import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score


def load_metrics(metrics_path: str) -> dict:
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding metrics JSON: {e}")
        sys.exit(1)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Params file not found: {params_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error decoding params YAML: {e}")
        sys.exit(1)


def recalculate_accuracy(model_path: str, test_data_path: str) -> float:
    try:
        # Загрузка модели
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Загрузка тестовых данных
        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop("target", axis=1)
        y_test = test_data["target"]
        
        # Пересчет accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return float(accuracy)
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error recalculating accuracy: {e}")
        sys.exit(1)


def validate_model(
    metrics_path: str,
    model_path: str,
    test_data_path: str,
    params_path: str,
    recalculate: bool = False
) -> bool:
    # Загрузка параметров
    params = load_params(params_path)
    accuracy_min = params.get("accuracy_min", 0.8)
    
    # Получение accuracy
    if recalculate:
        accuracy = recalculate_accuracy(model_path, test_data_path)
        print(f"Recalculated accuracy: {accuracy:.4f}")
    else:
        metrics = load_metrics(metrics_path)
        accuracy = metrics.get("accuracy", 0.0)
        print(f"Accuracy from metrics: {accuracy:.4f}")
    
    # Проверка порога
    print(f"Minimum required accuracy: {accuracy_min:.4f}")
    
    if accuracy >= accuracy_min:
        print("Model validation PASSED")
        return True
    else:
        print("Model validation FAILED")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model quality")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to metrics file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--params_path", type=str, required=True, help="Path to params file")
    parser.add_argument("--recalculate", action="store_true", help="Recalculate accuracy")
    
    args = parser.parse_args()
    
    is_valid = validate_model(
        metrics_path=args.metrics_path,
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        params_path=args.params_path,
        recalculate=args.recalculate
    )
    
    # Завершение с соответствующим кодом
    sys.exit(0 if is_valid else 1)
