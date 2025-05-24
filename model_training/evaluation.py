import joblib
import json
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from data_preprocessing import get_dataset


if __name__ == "__main__":
	model = joblib.load('models/naive_bayes.joblib')
	X_train, X_test, y_train, y_test = get_dataset()
	y_pred = model.predict(X_test)

	cm = confusion_matrix(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, output_dict=True)

	metrics = {
        	"accuracy": acc,
        	"precision": report["weighted avg"]["precision"],
        	"recall": report["weighted avg"]["recall"]}

	with open("metrics/evaluation_metrics.json", "w") as f:
		json.dump(metrics, f, indent=2)

	print("saved metrics: accuracy, precision, recall")
