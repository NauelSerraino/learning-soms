import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH_DATA_INPUT = os.path.join(ROOT_DIR, 'data', 'input')
PATH_DATA_OUTPUT = os.path.join(ROOT_DIR, 'data', 'output')

PATH_SRC = os.path.join(ROOT_DIR, 'src')

PATH_DEVELOP = os.path.join(ROOT_DIR, 'develop')