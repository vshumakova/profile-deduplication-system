CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS batches (
    batch_id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    object_key TEXT NOT NULL,
    s3_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    profile1 TEXT NOT NULL,
    profile2 TEXT NOT NULL,
    match_score DOUBLE PRECISION NOT NULL,
    is_duplicate BOOLEAN NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch_id
ON predictions(batch_id);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID NOT NULL,
    batch_id UUID NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL,
    cluster_size INTEGER NOT NULL,
    recommendation_mode TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_batch_id
ON clusters(batch_id);

CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id
ON clusters(cluster_id);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    first_seen_batch_id UUID NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    last_seen_batch_id UUID NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source_filename TEXT,

    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT,
    birthday TEXT,
    sex TEXT,
    created_at TEXT,

    non_processing_features TEXT,
    realtime_features TEXT,
    fs_features TEXT
);

CREATE INDEX IF NOT EXISTS idx_profiles_last_seen_batch_id
ON profiles(last_seen_batch_id);

CREATE INDEX IF NOT EXISTS idx_profiles_email
ON profiles(email);