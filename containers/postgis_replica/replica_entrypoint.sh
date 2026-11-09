#!/bin/bash
set -e

export PGDATA="/var/lib/postgresql/data/pgdata"

# Если база реплики еще не инициализирована, клонируем физический слепок с Primary через pg_basebackup
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "[REPLICA] Ожидание готовности узла postgis-primary..."
    until pg_isready -h postgis-primary -p 5432 -U vkr_admin; do
        sleep 2
    done

    echo "[REPLICA] Выполнение pg_basebackup с узла postgis-primary..."
    rm -rf "$PGDATA"/*
    PGPASSWORD=replicator_pass pg_basebackup \
        -h postgis-primary \
        -p 5432 \
        -U replicator \
        -D "$PGDATA" \
        -Fp -Xs -P -R

    chmod 0700 "$PGDATA"
    echo "[REPLICA] Базовый слепок успешно создан, режим Hot Standby (standby.signal) активирован."
fi

exec docker-entrypoint.sh postgres -c hot_standby=on
