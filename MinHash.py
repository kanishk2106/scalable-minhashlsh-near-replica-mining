from pyspark.sql.functions import array, col, udf, struct
from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector, SparseVector
from pyspark.sql.types import IntegerType, BooleanType, DoubleType, StructType, StructField
import os, sys, argparse
from urllib.request import urlopen
import time

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARN] pandas not available - charts will be skipped")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available - charts will be skipped")
def get_config(spark_session, key: str, default: str) -> str:
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        if spark_session is not None and hasattr(spark_session, "sparkContext"):
            sc = spark_session.sparkContext
            try:
                conf = sc.getConf()
            except Exception:
                conf = None
            if conf is not None:
                for full_key in (
                    key,
                    f"spark.yarn.appMasterEnv.{key}",
                    f"spark.driverEnv.{key}",
                    f"spark.executorEnv.{key}",
                ):
                    try:
                        val = conf.get(full_key)
                        if val is not None:
                            return val
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        if spark_session is not None:
            val = spark_session.conf.get(key)
            if val is not None:
                return val
            for full_key in (
                f"spark.yarn.appMasterEnv.{key}",
                f"spark.driverEnv.{key}",
                f"spark.executorEnv.{key}",
            ):
                try:
                    val = spark_session.conf.get(full_key)
                    if val is not None:
                        return val
                except Exception:
                    pass
    except Exception:
        pass

    return default

def safe_unpersist(df, blocking=True):
    try:
        if df is not None and hasattr(df, 'unpersist'):
            df.unpersist(blocking=blocking)
            return True
    except Exception as e:
        print(f"[WARN] Failed to unpersist DataFrame: {e}")
        return False
    return False

def _target_files(spark, floor=8, ceil=128, mult=3):
    try:
        base = spark.sparkContext.defaultParallelism
        k = max(floor, min(ceil, base * mult))
    except Exception:
        k = 16
    try:
        k = int(os.getenv("PARQUET_TARGET_FILES", str(k)))
    except Exception:
        pass
    return max(floor, min(ceil, k))

def _path_exists(spark, path: str) -> bool:
    try:
        jvm = spark._jvm
        conf = spark._jsc.hadoopConfiguration()
        fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI.create(path), conf)
        return fs.exists(jvm.org.apache.hadoop.fs.Path(path))
    except Exception:

        try:
            spark.read.parquet(path).limit(1).count()
            return True
        except Exception:
            return False

def model_path(base: str, nht: int) -> str:

    return f"{base}/minhash_n{nht}"

def load_or_fit_model(spark, df_features, nht: int, models_base: str, skip_fit: bool):
    mp = model_path(models_base, nht)
    try:
        print(f"[LSH] Trying to load n={nht} from {mp}")
        return MinHashLSHModel.load(mp)
    except Exception as e:
        if skip_fit:
            raise FileNotFoundError(
                f"[LSH] Missing model for n={nht} at {mp} and --skip_fit was set. "
                f"Run once without --skip_fit to fit+save."
            )
        print(f"[LSH] Fitting n={nht} (no existing model).")
        lsh = (MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=nht)
               .setSeed(42))
        model = lsh.fit(df_features)
        try:
            model.write().overwrite().save(mp)
            print(f"[LSH] Saved n={nht} to {mp}")
        except Exception as se:
            print(f"[WARN] Could not save model to {mp}: {se}")
        return model

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

spark = (
    SparkSession.builder
    .config("spark.sql.parquet.compression.codec", "snappy")

    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled", "true")
    .config("spark.sql.adaptive.localShuffleReader.enabled", "true")

    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "64")

    .config("spark.sql.files.maxPartitionBytes", "67108864")
    .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "512"))

    .getOrCreate()
)
sc = spark.sparkContext

print(f"[UI] Spark UI: {getattr(sc, 'uiWebUrl', None) or 'N/A'}")

try:
    shuffle_parts = int(spark.conf.get("spark.sql.shuffle.partitions", "512"))
    shuffle_parts = max(64, min(2048, shuffle_parts))
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_parts))
    print(f"[CONFIG] spark.sql.shuffle.partitions={shuffle_parts}")
except Exception:
    spark.conf.set("spark.sql.shuffle.partitions", "512")
    print(f"[CONFIG] spark.sql.shuffle.partitions=512 (default)")

try:
    broadcast_threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    if broadcast_threshold == "-1":
        print(f"[CONFIG] spark.sql.autoBroadcastJoinThreshold=-1 (broadcast disabled)")
    else:
        threshold_mb = int(broadcast_threshold) / (1024 * 1024)
        print(f"[CONFIG] spark.sql.autoBroadcastJoinThreshold={threshold_mb:.1f}MB")
except Exception:
    print(f"[CONFIG] spark.sql.autoBroadcastJoinThreshold=10MB (default)")

OUTPUT_BASE  = None
MODELS_BASE  = None
PARQUET_PATH = None
FORCE_PARQUET = True
DATA_FRACTION = 0.15

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--skip_fit", action="store_true",
                    help="Do NOT fit LSH models. Only load existing ones; error if missing.")
parser.add_argument("--only_grid", action="store_true",
                    help="Run only the grid (Steps 2–4 artifacts) and then exit.")
parser.add_argument("--pairs_storage", choices=["MEMORY_AND_DISK", "DISK_ONLY"], default="DISK_ONLY",
                    help="Storage level for pairs DataFrames during grid. Default DISK_ONLY (safer for skewed data).")

parser.add_argument("--parquet_path", type=str, help="Override PARQUET_PATH (e.g., gs://.../a2_df.parquet)")
parser.add_argument("--output_base", type=str, help="Override OUTPUT_BASE directory")
parser.add_argument("--models_base", type=str, help="Override MODELS_BASE directory")
parser.add_argument("--a3_base", type=str, help="Base path to derive OUTPUT/MODELS/PARQUET if others not provided")
parser.add_argument("--data_fraction", type=float, help="Override DATA_FRACTION (e.g., 0.15)")
parser.add_argument("--force_parquet", type=int, choices=[0,1], help="1 to force Parquet-only, 0 to allow CSV fallback")
args, _unknown = parser.parse_known_args()

if getattr(args, "a3_base", None):
    base = args.a3_base.rstrip("/")
    if not getattr(args, "output_base", None):
        OUTPUT_BASE = f"{base}/output"
    if not getattr(args, "models_base", None):
        MODELS_BASE = f"{base}/models"
    if not getattr(args, "parquet_path", None):
        PARQUET_PATH = f"{base}/a2_df.parquet"

if getattr(args, "parquet_path", None):
    PARQUET_PATH = args.parquet_path
if getattr(args, "output_base", None):
    OUTPUT_BASE = args.output_base
if getattr(args, "models_base", None):
    MODELS_BASE = args.models_base
if getattr(args, "data_fraction", None) is not None:
    DATA_FRACTION = float(args.data_fraction)
if getattr(args, "force_parquet", None) is not None:
    FORCE_PARQUET = bool(int(args.force_parquet))

if PARQUET_PATH and (OUTPUT_BASE is None or MODELS_BASE is None):
    p = PARQUET_PATH.rstrip("/")
    slash = p.rfind("/")
    if slash > len("gs://"):
        base = p[:slash]
        OUTPUT_BASE = OUTPUT_BASE or f"{base}/output"
        MODELS_BASE = MODELS_BASE or f"{base}/models"

if not PARQUET_PATH:
    raise SystemExit(
        "[FATAL] PARQUET_PATH not provided. Pass --parquet_path gs://... or --a3_base gs://..."
    )
if not OUTPUT_BASE:
    raise SystemExit(
        "[FATAL] OUTPUT_BASE not resolved. Pass --output_base gs://... or --a3_base gs://..."
    )
if not MODELS_BASE:
    raise SystemExit(
        "[FATAL] MODELS_BASE not resolved. Pass --models_base gs://... or --a3_base gs://..."
    )

print(f"[CONFIG] OUTPUT_BASE={OUTPUT_BASE}")
print(f"[CONFIG] MODELS_BASE={MODELS_BASE}")
print(f"[CONFIG] PARQUET_PATH={PARQUET_PATH}")
print(f"[CONFIG] FORCE_PARQUET={FORCE_PARQUET}")
print(f"[CONFIG] DATA_FRACTION={DATA_FRACTION}")

feature_cols = [str(i) for i in range(2048)]

if FORCE_PARQUET:
    if not _path_exists(spark, PARQUET_PATH):
        raise SystemExit(
            f"[FATAL] Parquet path not found: {PARQUET_PATH}\n"
            f"Upload it to HDFS or set FORCE_PARQUET=0 to allow CSV fallback."
        )
    print(f"Loaded DataFrame from Parquet (FORCE_PARQUET=1): {PARQUET_PATH}")
    df = spark.read.parquet(PARQUET_PATH)
else:

    if not _path_exists(spark, PARQUET_PATH):
        raise SystemExit(
            f"[FATAL] Parquet path not found: {PARQUET_PATH}\n"
            f"CSV fallback has been removed. Please provide a valid Parquet path.\n"
            f"Use --force_parquet 1 (default) or ensure your Parquet exists."
        )
    df = spark.read.parquet(PARQUET_PATH)
    print(f"Loaded DataFrame from Parquet: {PARQUET_PATH}")

required = {"SMILES", "SA_score", "SA_label"} | set(str(i) for i in range(2048))
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise SystemExit(f"[FATAL] Missing columns in PARQUET_PATH: {missing_cols[:10]}{'...' if len(missing_cols)>10 else ''}")

if df.rdd.isEmpty():
    raise SystemExit("[FATAL] Input DataFrame is empty; nothing to do.")

try:
    sample_col = "0"
    actual_dtype = df.schema[sample_col].dataType
    print(f"[DTYPE] Feature columns are {actual_dtype} (no transformation needed - assuming binary from Assignment 2)")
except Exception as e:
    print(f"[DTYPE] Warning: could not validate feature dtypes ({e}), proceeding anyway")

import pyspark.sql.functions as F

if 0.0 < DATA_FRACTION < 1.0:

    df = (df
          .withColumn("_bucket", F.pmod(F.xxhash64("SMILES"), F.lit(100)))
          .where(F.col("_bucket") < F.lit(int(DATA_FRACTION * 100)))
          .drop("_bucket"))
    print(f"DATA_FRACTION in effect: keeping ~{DATA_FRACTION*100:.1f}% of rows")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

to_sparse = udf(lambda v: v.toSparse() if isinstance(v, DenseVector) else v if isinstance(v, SparseVector) else None, VectorUDT())
df = df.withColumn("features", to_sparse("features"))

print("[ASSEMBLY] Materializing assembled features to prevent execution plan overflow")
temp_assembly_path = f"{OUTPUT_BASE}/temp_assembled_features.parquet"
if not _path_exists(spark, temp_assembly_path):
    _k = _target_files(spark)

    cols_to_keep = ["SMILES", "features", "SA_score", "SA_label"] + [c for c in df.columns if c in ["id"]]
    (df.select(*[c for c in cols_to_keep if c in df.columns])
       .repartition(_k)
       .write
       .mode("overwrite")
       .option("compression", "snappy")
       .parquet(temp_assembly_path))
    print(f"[ASSEMBLY] Wrote assembled features to {temp_assembly_path}")
else:
    print(f"[ASSEMBLY] Using existing assembled features from {temp_assembly_path}")

df = spark.read.parquet(temp_assembly_path)
print("[ASSEMBLY] Reloaded assembled DataFrame with fresh lineage")

@udf(returnType=BooleanType())
def has_nonzero_features(vec):
    """Check if vector has at least one non-zero element"""
    if vec is None:
        return False

    if hasattr(vec, 'indices') and vec.indices is not None:
        return len(vec.indices) > 0

    arr = vec.toArray()
    return any(arr[i] != 0 for i in range(len(arr)))

rows_before = df.count()
df = df.filter(has_nonzero_features(F.col("features")))
rows_after = df.count()
removed = rows_before - rows_after

if removed > 0:
    print(f"[FILTER] Removed {removed} rows with all-zero fingerprints ({removed/rows_before*100:.2f}%)")
else:
    print(f"[FILTER] No all-zero fingerprints found (good - all {rows_after} molecules have structural features)")

print(f"Total rows after filtering: {rows_after}")
df.select("features").printSchema()

minhash_models = {}

import pyspark.sql.functions as F
from pyspark.storagelevel import StorageLevel

if "id" not in df.columns:
    try:
        df = df.withColumn("id", F.xxhash64("SMILES"))
    except Exception:
        df = df.withColumn("id", F.crc32("SMILES").cast("long"))

print("[OPTIMIZATION] Creating id->SMILES lookup map and minimal LSH working set")

id_smiles_map = df.select("id", "SMILES").persist(StorageLevel.MEMORY_AND_DISK)
map_count = id_smiles_map.count()
print(f"[OPTIMIZATION] Created id->SMILES map with {map_count} entries (~{map_count * 58 / 1024 / 1024:.1f} MB)")

df_proj_base = df.select("id", "features")

USE_CHECKPOINT = int(os.getenv("USE_CHECKPOINT", "1")) == 1

if USE_CHECKPOINT and OUTPUT_BASE:
    try:

        checkpoint_dir = get_config(spark, "CHECKPOINT_DIR", None)
        if not checkpoint_dir:
            checkpoint_dir = f"{OUTPUT_BASE}/checkpoint"

        if checkpoint_dir:
            spark.sparkContext.setCheckpointDir(checkpoint_dir)
            print(f"[CHECKPOINT] Using reliable checkpoint dir: {checkpoint_dir}")

            df_proj_base = df_proj_base.checkpoint(eager=True)
            print(f"[CHECKPOINT] Successfully checkpointed slim DataFrame (2 cols: id, features)")
        else:
            raise ValueError("CHECKPOINT_DIR is empty")
    except Exception as e:
        print(f"[CHECKPOINT] Warning: checkpoint failed ({e}), using DISK_ONLY persist instead")
        df_proj_base = df_proj_base.persist(StorageLevel.DISK_ONLY)
        _ = df_proj_base.count()
else:
    if not OUTPUT_BASE:
        print("[CHECKPOINT] Warning: OUTPUT_BASE missing, cannot checkpoint. Using DISK_ONLY persist.")
    else:
        print("[CHECKPOINT] Skipping checkpoint (USE_CHECKPOINT=0), using DISK_ONLY persist")
    df_proj_base = df_proj_base.persist(StorageLevel.DISK_ONLY)
    _ = df_proj_base.count()

try:
    _nparts = int(spark.conf.get("spark.sql.shuffle.partitions", "512"))
    _nparts = max(64, min(2048, _nparts))
except Exception:
    _nparts = 512
df_proj_base = df_proj_base.repartition(_nparts, F.xxhash64("id"))

dfA = df_proj_base.alias("datasetA")
dfB = df_proj_base.alias("datasetB")

assembled_out = f"{OUTPUT_BASE}/assembled/df_proj_base.parquet"
_k = _target_files(spark)

if not _path_exists(spark, assembled_out):
    (df_proj_base
        .repartition(_k)
        .write
        .mode("ignore")
        .option("compression", "snappy")
        .option("parquet.block.size", 134217728)
        .option("maxRecordsPerFile", 5_000_000)
        .parquet(assembled_out))

df_features = df_proj_base.select("features").persist(StorageLevel.DISK_ONLY)
_ = df_features.count()

minhash_models = {}
nht_list = [16, 32, 64]
missing = []
for nht in nht_list:
    try:
        minhash_models[nht] = load_or_fit_model(spark, df_features, nht, MODELS_BASE, bool(getattr(args, 'skip_fit', False)))
    except FileNotFoundError as e:
        print(str(e))
        missing.append(nht)

if missing and bool(getattr(args, 'skip_fit', False)):
    print(f"[FATAL] Missing models with --skip_fit: {missing}. Run once without --skip_fit to fit+save.")
    sys.exit(2)

df_features.unpersist()
print("[CLEANUP] Unpersisted df_features after model loading (no longer needed)")

taus = [0.80, 0.90, 0.95]
metrics_rows = []

existing_metrics_path = f"{OUTPUT_BASE}/metrics/lsh_grid_metrics.csv"
if _path_exists(spark, existing_metrics_path):
    try:
        print(f"[RESUME] Loading existing metrics from {existing_metrics_path}")
        existing_df = spark.read.option("header", True).csv(existing_metrics_path)
        existing_rows = existing_df.collect()
        for row in existing_rows:
            metrics_rows.append({
                "nht": int(row['nht']),
                "tau": float(row['tau']),
                "num_pairs": int(row['num_pairs']),
                "runtime_sec": float(row['runtime_sec']),
                "shuffle_read_bytes": int(row['shuffle_read_bytes']) if row['shuffle_read_bytes'] else 0,
                "shuffle_write_bytes": int(row['shuffle_write_bytes']) if row['shuffle_write_bytes'] else 0,
            })
        print(f"[RESUME] Loaded {len(existing_rows)} existing metric rows")
    except Exception as e:
        print(f"[WARN] Could not load existing metrics: {e}")
        print("[WARN] Starting fresh metrics collection")

print("METRICS,nht,tau,num_pairs,runtime_sec,shuffle_read_bytes,shuffle_write_bytes")

def _fetch_shuffle_metrics_for_group(sc, group_id):
    try:
        ui = getattr(sc, "uiWebUrl", None)
        ui_web_url = ui if isinstance(ui, str) else (ui() if callable(ui) else None)
        app_id = sc.applicationId
        if not ui_web_url or not app_id:
            return None, None

        import json

        def _get_json(url):
            try:
                with urlopen(url, timeout=2.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                return None

        job_ids = sc.statusTracker().getJobIdsForGroup(group_id) or []
        base = ui_web_url.rstrip("/")
        total_read = 0
        total_write = 0
        saw_any = False

        def _sum_stage(stage_obj):
            sr = 0
            sw = 0
            try:
                sr += int(stage_obj.get("shuffleReadBytes", 0) or 0)
            except Exception:
                pass
            try:
                sw += int(stage_obj.get("shuffleWriteBytes", 0) or 0)
            except Exception:
                pass
            m = stage_obj.get("metrics") or stage_obj.get("taskMetrics") or {}
            if isinstance(m, dict):
                srm = m.get("shuffleReadMetrics") or {}
                swm = m.get("shuffleWriteMetrics") or {}
                try:
                    sr += int(srm.get("totalBytesRead", 0) or 0)
                except Exception:
                    pass
                try:
                    sw += int(swm.get("bytesWritten", 0) or 0)
                except Exception:
                    pass
            ex = stage_obj.get("executorSummary") or stage_obj.get("executorSummaryMap") or {}
            if isinstance(ex, dict):
                for v in ex.values():
                    srm = (v.get("shuffleRead") or {})
                    swm = (v.get("shuffleWrite") or {})
                    try:
                        sr += int(srm.get("totalBytesRead", 0) or 0)
                    except Exception:
                        pass
                    try:
                        sw += int(swm.get("bytesWritten", 0) or 0)
                    except Exception:
                        pass
            return sr, sw

        for jid in job_ids:
            job_url = f"{base}/api/v1/applications/{app_id}/jobs/{jid}"
            job_json = _get_json(job_url)
            if not job_json:
                continue
            stage_ids = job_json.get("stageIds") or []
            for sid in stage_ids:
                stage_url = f"{base}/api/v1/applications/{app_id}/stages/{sid}"
                stage_json = _get_json(stage_url)
                if not stage_json:
                    continue
                stage_obj = stage_json[-1] if isinstance(stage_json, list) and stage_json else (
                    stage_json if isinstance(stage_json, dict) else None
                )
                if not stage_obj:
                    continue
                sr, sw = _sum_stage(stage_obj)
                total_read += sr
                total_write += sw
                saw_any = True

        if not saw_any:
            return (0, 0)
        return (int(total_read), int(total_write))
    except Exception:
        return (0, 0)

from pyspark.sql.functions import col as _col

pairs_storage_arg = getattr(args, 'pairs_storage', 'DISK_ONLY')
if pairs_storage_arg == 'DISK_ONLY':
    storage = StorageLevel.DISK_ONLY
else:
    storage = StorageLevel.MEMORY_AND_DISK_SER

for nht in (16, 32, 64):

    grid_pairs_path = f"{OUTPUT_BASE}/grid_lsh_pairs/nht_{nht}/pairs.parquet"
    if _path_exists(spark, grid_pairs_path):
        print(f"\n[SKIP] Grid pairs already exist for nht={nht}: {grid_pairs_path}")
        print(f"[SKIP] Skipping LSH computation for nht={nht} (reusing existing results)")
        continue

    model = minhash_models[nht]

    tau_min = min(taus)
    radius_max = 1.0 - tau_min

    print(f"\n[LSH] Running single join for nht={nht} with tau_min={tau_min} (will filter for higher tau)")

    group_id = f"lsh_nht_{nht}_base_join"
    spark.sparkContext.setJobGroup(group_id, f"LSH approxSimilarityJoin for nht={nht}, tau={tau_min} (base)")

    joined = model.approxSimilarityJoin(dfA, dfB, radius_max, distCol="JaccardDistance")

    pairs_base = (joined
             .where(_col("datasetA.id") < _col("datasetB.id"))
             .repartition(max(1, _nparts // 2), F.xxhash64(_col("datasetA.id"), _col("datasetB.id")))
             .persist(storage))

    base_start = time.time()
    _ = pairs_base.count()
    base_runtime = time.time() - base_start
    print(f"[LSH] Base join for nht={nht} materialized in {base_runtime:.1f}s")

    for tau in sorted(taus):
        print(f"[LSH] Processing nht={nht}, tau={tau} (filtering base pairs)")
        group_id = f"lsh_nht_{nht}_tau_{tau}"
        spark.sparkContext.setJobGroup(group_id, f"LSH filter for nht={nht}, tau={tau}")

        if tau == tau_min:

            pairs = pairs_base
            runtime_sec = base_runtime
        else:

            pairs = pairs_base.where(_col("JaccardDistance") <= (1.0 - tau))
            runtime_sec = 0.0

        start = time.time()
        try:
            num_pairs = pairs.count()
            count_time = time.time() - start
            if tau != tau_min:
                runtime_sec = count_time

            ui = getattr(spark.sparkContext, "uiWebUrl", None)
            ui_url = ui if isinstance(ui, str) else (ui() if callable(ui) else None)
            if ui_url:
                shuffle_read_bytes, shuffle_write_bytes = _fetch_shuffle_metrics_for_group(spark.sparkContext, group_id)
            else:
                shuffle_read_bytes, shuffle_write_bytes = (0, 0)

            if shuffle_read_bytes is None:
                shuffle_read_bytes = 0
            if shuffle_write_bytes is None:
                shuffle_write_bytes = 0

            print(f"METRICS,{nht},{tau},{num_pairs},{round(runtime_sec, 3)},{shuffle_read_bytes},{shuffle_write_bytes}")

            tau_no_dot = f"{int(tau * 100):03d}"
            sample_path = f"{OUTPUT_BASE}/samples/nht_{nht}/tau_{tau_no_dot}/sample.parquet"
            sample_df = (
                pairs
                .select(
                    _col("datasetA.id").alias("id_a"),
                    _col("datasetB.id").alias("id_b"),
                    (1.0 - _col("JaccardDistance")).alias("approx_tanimoto"),
                    _col("JaccardDistance")
                )
                .orderBy(_col("approx_tanimoto").desc())
                .limit(25)

                .join(F.broadcast(id_smiles_map.alias("map_a")), _col("id_a") == _col("map_a.id"), "left")
                .join(F.broadcast(id_smiles_map.alias("map_b")), _col("id_b") == _col("map_b.id"), "left")
                .select(
                    _col("map_a.SMILES").alias("SMILES_a"),
                    _col("map_b.SMILES").alias("SMILES_b"),
                    _col("approx_tanimoto"),
                    _col("JaccardDistance")
                )
                .coalesce(1)
            )
            sample_df.write.mode("overwrite").parquet(sample_path)
            print(f"[SAMPLE] nht={nht}, tau={tau} -> {sample_path}")

            metrics_rows.append({
                "nht": int(nht),
                "tau": float(tau),
                "num_pairs": int(num_pairs),
                "runtime_sec": round(float(runtime_sec), 3),
                "shuffle_read_bytes": int(shuffle_read_bytes) if shuffle_read_bytes is not None else None,
                "shuffle_write_bytes": int(shuffle_write_bytes) if shuffle_write_bytes is not None else None,
            })
        finally:
            try:
                sc = spark.sparkContext
                if hasattr(sc, "clearJobGroup"):
                    sc.clearJobGroup()
                else:
                    sc.setLocalProperty("spark.jobGroup.id", None)
            except Exception:
                pass

    grid_pairs_path = f"{OUTPUT_BASE}/grid_lsh_pairs/nht_{nht}/pairs.parquet"
    print(f"[OPTIMIZATION] Saving grid pairs to {grid_pairs_path}")

    (pairs_base
        .select(
            _col("datasetA.id").alias("id_a"),
            _col("datasetB.id").alias("id_b"),
            _col("JaccardDistance")
        )
        .write
        .mode("overwrite")
        .parquet(grid_pairs_path))

    print(f"[OPTIMIZATION] Grid pairs saved for nht={nht}")

    pairs_base.unpersist()
    print(f"[CLEANUP] Unpersisted base pairs for nht={nht} (saved to Parquet)")

if metrics_rows:

    seen_keys = {}
    for row in metrics_rows:
        key = (row['nht'], row['tau'])
        seen_keys[key] = row

    deduplicated_rows = list(seen_keys.values())
    print(f"[METRICS] Collected {len(metrics_rows)} rows, deduplicated to {len(deduplicated_rows)} unique (nht, tau) pairs")

    summary_df = spark.createDataFrame(deduplicated_rows)
else:
    empty_schema = StructType([
        StructField("nht", IntegerType(), True),
        StructField("tau", DoubleType(), True),
        StructField("num_pairs", IntegerType(), True),
        StructField("runtime_sec", DoubleType(), True),
        StructField("shuffle_read_bytes", IntegerType(), True),
        StructField("shuffle_write_bytes", IntegerType(), True),
    ])
    summary_df = spark.createDataFrame([], empty_schema)
    print("[WARN] No metrics collected; writing empty summary")

summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_BASE}/metrics/lsh_grid_metrics.csv")

if bool(getattr(args, 'only_grid', False)):
    sys.exit(0)

print("\n=== STEP 5: Evaluation vs Exact Tanimoto ===")

total_count = df_proj_base.count()

if total_count == 0:
    raise SystemExit("[FATAL] No rows after projection; cannot evaluate.")

MAX_EVAL_ROWS = int(os.getenv("MAX_EVAL_ROWS", "50000"))

print(f"[EVAL] MAX_EVAL_ROWS={MAX_EVAL_ROWS}")
if MAX_EVAL_ROWS > 10_000:
    est_pairs = (MAX_EVAL_ROWS * (MAX_EVAL_ROWS - 1)) // 2
    print(
        f"[WARN] MAX_EVAL_ROWS={MAX_EVAL_ROWS} may generate up to ~{est_pairs:,} "
        "exact Tanimoto pairs (O(N^2) evaluation). Ensure your cluster is large enough."
    )
if MAX_EVAL_ROWS > 50_000:
    raise SystemExit(
        f"[FATAL] MAX_EVAL_ROWS={MAX_EVAL_ROWS} is too large for this pipeline. "
        "Set MAX_EVAL_ROWS<=50000 or reduce DATA_FRACTION."
    )

SAMPLE_SIZE = min(total_count, MAX_EVAL_ROWS)
SAMPLE_SEED = 42

print(f"[EVAL] Using {SAMPLE_SIZE} molecules (from {total_count} total) for Tanimoto ground truth")
print(f"[EVAL] This will generate ~{(SAMPLE_SIZE * (SAMPLE_SIZE - 1)) // 2:,} pairwise comparisons")

fraction = min(1.0, max(0.0, SAMPLE_SIZE / total_count))
if fraction == 0.0:
    print("[EVAL] Sample fraction is 0; skipping evaluation step.")
    sys.exit(0)

if fraction < 1.0:
    df_sample = df_proj_base.sample(withReplacement=False, fraction=fraction, seed=SAMPLE_SEED)
else:
    df_sample = df_proj_base

sample_count = df_sample.count()
print(f"Sampled {sample_count} molecules for exact Tanimoto evaluation")

print("[OPTIMIZATION] Computing popcount for each molecule (cheap for sparse vectors)")

def get_popcount(sparse_vec):
    """Get number of 1s in a sparse vector (length of indices array)"""
    if sparse_vec is None:
        return 0
    if hasattr(sparse_vec, 'indices') and sparse_vec.indices is not None:
        return len(sparse_vec.indices)

    return int(sum(1 for x in sparse_vec.toArray() if x > 0))

popcount_udf = udf(get_popcount, IntegerType())

df_sample = df_sample.withColumn("popcount", popcount_udf(_col("features")))

df_sample = df_sample.persist(StorageLevel.MEMORY_AND_DISK)

print(f"[EVAL OPTIMIZATION] Collecting {sample_count} sample IDs for grid pair filtering")
sample_ids_list = df_sample.select("id").rdd.flatMap(lambda x: x).collect()
sample_ids_set = set(sample_ids_list)
print(f"[EVAL OPTIMIZATION] Collected {len(sample_ids_set)} unique sample IDs")

def exact_tanimoto(vec1, vec2):
    """Compute exact Tanimoto similarity between two binary vectors"""
    if vec1 is None or vec2 is None:
        return None

    if (hasattr(vec1, 'indices') and hasattr(vec2, 'indices') and
        vec1.indices is not None and vec2.indices is not None):

        try:
            set1 = set(vec1.indices.tolist() if hasattr(vec1.indices, 'tolist') else vec1.indices)
            set2 = set(vec2.indices.tolist() if hasattr(vec2.indices, 'tolist') else vec2.indices)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
        except Exception:

            arr1 = vec1.toArray() if hasattr(vec1, 'toArray') else vec1
            arr2 = vec2.toArray() if hasattr(vec2, 'toArray') else vec2
            intersection = sum((arr1[i] > 0 and arr2[i] > 0) for i in range(len(arr1)))
            union = sum((arr1[i] > 0 or arr2[i] > 0) for i in range(len(arr1)))
    else:

        arr1 = vec1.toArray() if hasattr(vec1, 'toArray') else vec1
        arr2 = vec2.toArray() if hasattr(vec2, 'toArray') else vec2
        intersection = sum((arr1[i] > 0 and arr2[i] > 0) for i in range(len(arr1)))
        union = sum((arr1[i] > 0 or arr2[i] > 0) for i in range(len(arr1)))

    if union == 0:
        return 0.0
    return float(intersection) / float(union)

exact_tanimoto_udf = udf(exact_tanimoto, DoubleType())

eval_results = []

exact_pairs_path = f"{OUTPUT_BASE}/exact_tanimoto_pairs/sample_{sample_count}/pairs.parquet"

if _path_exists(spark, exact_pairs_path):
    print(f"\n[RESUME] Exact Tanimoto pairs already exist: {exact_pairs_path}")
    print(f"[RESUME] Loading from disk instead of recomputing cross-join (huge time saver!)")
    exact_pairs_all = spark.read.parquet(exact_pairs_path)
    total_exact_pairs = exact_pairs_all.count()
    print(f"[EVAL] Loaded {total_exact_pairs:,} exact pairwise similarities from Parquet")
else:
    print(f"\n[EVAL OPTIMIZATION] Computing exact Tanimoto ONCE for all {sample_count} molecules")
    print(f"[EVAL] This will generate ~{(sample_count * (sample_count - 1)) // 2:,} pairwise comparisons")

    exact_a = df_sample.alias("a")
    exact_b = df_sample.alias("b")

    num_partitions = max(32, min(256, sample_count // 200))
    exact_a = exact_a.repartition(num_partitions)
    exact_b = exact_b.repartition(num_partitions)

    tau_min = min(taus)
    print(f"[OPTIMIZATION] Using popcount filter: only compute Tanimoto for pairs with upper_bound >= {tau_min}")

    exact_pairs_all = (
        exact_a.crossJoin(exact_b)
               .where(_col("a.id") < _col("b.id"))

               .withColumn("max_intersection", F.least(_col("a.popcount"), _col("b.popcount")))
               .withColumn("min_union", F.greatest(_col("a.popcount"), _col("b.popcount")))
               .withColumn("upper_bound",
                   F.when(_col("min_union") > 0, _col("max_intersection") / _col("min_union"))
                    .otherwise(0.0))
               .where(_col("upper_bound") >= tau_min)
               .withColumn(
                   "exact_tanimoto",
                   exact_tanimoto_udf(_col("a.features"), _col("b.features"))
               )
               .select(
                   _col("a.id").alias("id_a"),
                   _col("b.id").alias("id_b"),
                   _col("exact_tanimoto")
               )
               .repartition(num_partitions * 2, F.xxhash64("id_a", "id_b"))
    )

    print(f"[OPTIMIZATION] Popcount filter will skip pairs with dissimilar bit counts")

    print(f"[OPTIMIZATION] Saving exact Tanimoto pairs to {exact_pairs_path}")
    print(f"[OPTIMIZATION] This prevents memory overflow and enables resume if job crashes")

    exact_pairs_all.write.mode("overwrite").parquet(exact_pairs_path)

    exact_pairs_all = spark.read.parquet(exact_pairs_path)

    total_exact_pairs = exact_pairs_all.count()
    max_possible_pairs = (sample_count * (sample_count - 1)) // 2
    filtered_out = max_possible_pairs - total_exact_pairs
    filter_percent = (filtered_out / max_possible_pairs * 100) if max_possible_pairs > 0 else 0
    print(f"[EVAL] Computed and saved {total_exact_pairs:,} exact pairwise similarities to Parquet")
    print(f"[OPTIMIZATION] Popcount filter eliminated {filtered_out:,} pairs ({filter_percent:.1f}%) - "
          f"UDF only called {total_exact_pairs:,} times instead of {max_possible_pairs:,}")

for tau in taus:
    print(f"\nEvaluating tau={tau}")

    exact_pairs_at_tau = exact_pairs_all.where(_col("exact_tanimoto") >= tau).persist(storage)
    true_replicas = exact_pairs_at_tau.count()
    print(f"  True replicas at tau={tau}: {true_replicas:,}")

    for nht in nht_list:

        group_id = f"lsh_eval_nht_{nht}_tau_{tau}"
        spark.sparkContext.setJobGroup(group_id, f"LSH eval for nht={nht}, tau={tau}")

        grid_pairs_path = f"{OUTPUT_BASE}/grid_lsh_pairs/nht_{nht}/pairs.parquet"
        print(f"[EVAL OPTIMIZATION] Reading grid pairs from {grid_pairs_path}")

        grid_pairs = spark.read.parquet(grid_pairs_path)

        lsh_pair_ids_only = grid_pairs.filter(
            (_col("id_a").isin(sample_ids_set)) &
            (_col("id_b").isin(sample_ids_set)) &
            (_col("JaccardDistance") <= (1.0 - tau))
        )

        print(f"[EVAL OPTIMIZATION] Filtered grid pairs for {len(sample_ids_set)} sample IDs, tau={tau}")

        lsh_pair_ids = lsh_pair_ids_only.select(
            F.concat_ws("_", _col("id_a"), _col("id_b")).alias("pair_id")
        ).persist(storage)

        lsh_count = lsh_pair_ids.count()
        print(f"[EVAL] LSH found {lsh_count:,} pairs for nht={nht}, tau={tau}")

        exact_with_id = exact_pairs_at_tau.select(
            F.concat_ws("_", _col("id_a"), _col("id_b")).alias("pair_id")
        )

        true_positives = lsh_pair_ids.join(exact_with_id, "pair_id", "inner").count()

        precision = true_positives / lsh_count if lsh_count > 0 else 0.0

        recall = true_positives / true_replicas if true_replicas > 0 else 0.0

        print(f"  nht={nht}: Precision={precision:.4f}, Recall={recall:.4f}, "
              f"LSH_pairs={lsh_count}, True_replicas={true_replicas}, True_positives={true_positives}")

        eval_results.append({
            "tau": float(tau),
            "nht": int(nht),
            "precision": float(precision),
            "recall": float(recall),
            "lsh_pairs": int(lsh_count),
            "true_replicas": int(true_replicas),
            "true_positives": int(true_positives),
            "sample_size": int(sample_count)
        })

        try:
            sc = spark.sparkContext
            if hasattr(sc, "clearJobGroup"):
                sc.clearJobGroup()
            else:
                sc.setLocalProperty("spark.jobGroup.id", None)
        except Exception:
            pass

        lsh_pair_ids.unpersist()

    exact_pairs_at_tau.unpersist()

print("[CLEANUP] Exact Tanimoto pairs remain saved in Parquet for future reuse")

eval_df = spark.createDataFrame(eval_results)
eval_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_BASE}/metrics/evaluation_metrics.csv")
print(f"\nEvaluation results saved to {OUTPUT_BASE}/metrics/evaluation_metrics.csv")
df_sample.unpersist()
print("\n=== STEP 6: Scaling Study ===")
print("[INFO] Chart generation skipped - all metrics saved to CSVs")
print("\n=== STEP 7: Bio-Relevant Insight - Deduplication ===")
dedup_results = []
grid_metrics_pd_local = spark.createDataFrame(metrics_rows).toPandas()
for tau in taus:

    nht = 64

    existing = grid_metrics_pd_local[(grid_metrics_pd_local['nht'] == nht) &
                                      (grid_metrics_pd_local['tau'] == tau)]

    if not existing.empty:

        print(f"\nDeduplication analysis at tau={tau} (nht={nht}) - reusing grid metrics")
        num_dup_pairs = int(existing.iloc[0]['num_pairs'])
        total_molecules = df_proj_base.count()

        dedup_fraction = num_dup_pairs / total_molecules if total_molecules > 0 else 0.0

        print(f"  Total molecules: {total_molecules}")
        print(f"  Near-replica pairs: {num_dup_pairs} (from grid)")
        print(f"  Dedup fraction (pairs/molecules): {dedup_fraction:.4f}")
        print(f"  Note: To get exact duplicate count, use graph-based connected components")

        tau_no_dot = f"{int(tau * 100):03d}"
        sample_path = f"{OUTPUT_BASE}/samples/nht_{nht}/tau_{tau_no_dot}/sample.parquet"
        print(f"  Top duplicates already saved: {sample_path}")

        dedup_results.append({
            "tau": float(tau),
            "total_molecules": int(total_molecules),
            "near_replica_pairs": int(num_dup_pairs),
            "dedup_fraction": float(dedup_fraction)
        })
    else:

        print(f"\nDeduplication analysis at tau={tau} (nht={nht}) - running LSH")
        model = minhash_models[nht]

        group_id = f"dedup_tau_{tau}"
        spark.sparkContext.setJobGroup(group_id, f"Deduplication at tau={tau}")

        radius = 1.0 - tau
        datasetA = df_proj_base.alias("datasetA")
        datasetB = df_proj_base.alias("datasetB")

        dup_pairs = model.approxSimilarityJoin(datasetA, datasetB, radius, distCol="JaccardDistance")

        dup_pairs = (dup_pairs
                     .where(_col("datasetA.id") < _col("datasetB.id"))
                     .repartition(max(1, _nparts // 2), F.xxhash64(_col("datasetA.id"), _col("datasetB.id")))
                     .persist(storage))

        num_dup_pairs = dup_pairs.count()
        total_molecules = df_proj_base.count()

        dedup_fraction = num_dup_pairs / total_molecules if total_molecules > 0 else 0.0

        print(f"  Total molecules: {total_molecules}")
        print(f"  Near-replica pairs: {num_dup_pairs}")
        print(f"  Dedup fraction (pairs/molecules): {dedup_fraction:.4f}")
        print(f"  Note: To get exact duplicate count, use graph-based connected components")

        top_dups = (dup_pairs
                    .select(
                        _col("datasetA.id").alias("id_a"),
                        _col("datasetB.id").alias("id_b"),
                        (1.0 - _col("JaccardDistance")).alias("similarity")
                    )
                    .orderBy(_col("similarity").desc())
                    .limit(100)

                    .join(F.broadcast(id_smiles_map.alias("map_a")), _col("id_a") == _col("map_a.id"), "left")
                    .join(F.broadcast(id_smiles_map.alias("map_b")), _col("id_b") == _col("map_b.id"), "left")
                    .select(
                        _col("map_a.SMILES").alias("SMILES_a"),
                        _col("map_b.SMILES").alias("SMILES_b"),
                        _col("similarity")
                    ))

        tau_no_dot = f"{int(tau * 100):03d}"
        dedup_path = f"{OUTPUT_BASE}/deduplication/tau_{tau_no_dot}/top_duplicates.parquet"
        top_dups.coalesce(1).write.mode("overwrite").parquet(dedup_path)

        dedup_results.append({
            "tau": float(tau),
            "total_molecules": int(total_molecules),
            "near_replica_pairs": int(num_dup_pairs),
            "dedup_fraction": float(dedup_fraction)
        })

        try:
            sc = spark.sparkContext
            if hasattr(sc, "clearJobGroup"):
                sc.clearJobGroup()
            else:
                sc.setLocalProperty("spark.jobGroup.id", None)
        except Exception:
            pass
        dup_pairs.unpersist()

dedup_df = spark.createDataFrame(dedup_results)
dedup_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_BASE}/metrics/deduplication_metrics.csv")
print(f"\nDeduplication results saved to {OUTPUT_BASE}/metrics/deduplication_metrics.csv")