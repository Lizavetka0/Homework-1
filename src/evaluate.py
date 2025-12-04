import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_model(model_path: str, test_data_path: str, metrics_path: str) -> None:
    # Загрузка модели
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Загрузка тестовых данных
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_pred)
    num_samples = len(test_data)
    
    # Создание директории для метрик 
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Сохранение метрик
    metrics = {
        "accuracy": float(accuracy),
        "num_samples": int(num_samples)
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of samples: {num_samples}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model and save metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to save metrics")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        metrics_path=args.metrics_path
    )
