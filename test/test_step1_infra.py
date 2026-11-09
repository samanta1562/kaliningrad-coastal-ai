import os
import yaml

def test_project_directories_exist():
    """Проверка создания структуры каталогов Data Lakehouse."""
    assert os.path.isdir("data_lake/bronze"), "Папка data_lake/bronze не найдена"
    assert os.path.isdir("data_lake/silver"), "Папка data_lake/silver не найдена"
    assert os.path.isdir("containers/etl_ingestor/src"), "Папка исходников Контейнера 1 не найдена"

def test_docker_compose_container1_config():
    """Проверка конфигурации Контейнера 1, лимитов памяти (под 8 ГБ RAM) и сети."""
    assert os.path.isfile("docker-compose.yml"), "Файл docker-compose.yml отсутствует"
    
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)
        
    assert "etl-ingestor" in compose["services"], "Сервис etl-ingestor не описан"
    service = compose["services"]["etl-ingestor"]
    
    # Проверяем ограничение памяти и сеть
    assert service.get("mem_limit") == "1024m", "Не задан лимит памяти 1024m"
    assert "backend_net" in service.get("networks", []), "Контейнер не подключен к backend_net"
    assert service["environment"][1] == "RBAC_ROLE=etl_ingestor", "Не задана роль RBAC"


