# from datetime import datetime
# from src.app.schemas import Prediction
# from src.app.aggregation.aggregation import aggregate_predictions

# test_predictions = [
#     Prediction(route_id=1, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=10),
#     Prediction(route_id=2, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=15),

#     Prediction(route_id=3, office_from_id=1, timestamp=datetime(2026, 3, 28, 10, 30), predicted_target_2h=20),

#     Prediction(route_id=4, office_from_id=2, timestamp=datetime(2026, 3, 28, 10, 0), predicted_target_2h=8),
# ]

# expected_result = [
#     {"office_from_id": 1, "timestamp": datetime(2026, 3, 28, 10, 0), "total_volume": 25},
#     {"office_from_id": 1, "timestamp": datetime(2026, 3, 28, 10, 30), "total_volume": 20},
#     {"office_from_id": 2, "timestamp": datetime(2026, 3, 28, 10, 0), "total_volume": 8},
# ]

# result = aggregate_predictions(test_predictions)

# if result == expected_result:
#     print("✅ Тест пройден!")
# else:
#     print("❌ Тест провален!")
#     print("Ожидалось:", expected_result)
#     print("Получено:", result)