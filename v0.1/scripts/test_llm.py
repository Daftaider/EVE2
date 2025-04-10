import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_file_operations():
    # Get the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'llm', 'simple_model.json')
    
    logger.info(f"Testing file operations for: {model_path}")
    
    # Test directory creation
    model_dir = os.path.dirname(model_path)
    try:
        os.makedirs(model_dir, mode=0o777, exist_ok=True)
        logger.info("Directory created/exists")
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return
    
    # Test file writing
    try:
        test_data = {"test": "data"}
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        logger.info("Successfully wrote test file")
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return
    
    # Test file reading
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Successfully read test file")
        logger.info(f"Content: {data}")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    logger.info("All file operations successful")

if __name__ == "__main__":
    test_file_operations() 