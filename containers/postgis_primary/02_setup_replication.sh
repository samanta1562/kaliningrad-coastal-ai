#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'replicator') THEN
            CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'replicator_pass';
        END IF;
    END
    \$\$;
EOSQL

# Разрешаем потоковую репликацию внутри подсети Docker
echo "host replication replicator 0.0.0.0/0 md5" >> "$PGDATA/pg_hba.conf"
