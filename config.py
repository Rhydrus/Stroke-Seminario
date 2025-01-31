from pathlib import Path
import os
import sys
PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd().parent
# Rutas comunes
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw_data'
# Añadir la ruta del proyecto al PYTHONPATH
sys.path.append(str(PROJECT_ROOT))

# Verificar el PYTHONPATH
print("PYTHONPATH:", sys.path)