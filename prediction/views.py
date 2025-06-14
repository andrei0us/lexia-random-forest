# views.py (in your Django app, e.g. prediction/views.py)
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
import pymysql
from django.db import connection

@csrf_exempt
def predict_performance(request):
    # Step 1: Load data from your MySQL database table: student_cluster_data
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT student_id, avg_accuracy, avg_consistency, avg_speed,
                   avg_retention, avg_problem_solving_skills, avg_vocabulary_range,
                   fk_section_id
            FROM student_cluster_data
        """)
        rows = cursor.fetchall()

    columns = [
        'student_id', 'avg_accuracy', 'avg_consistency', 'avg_speed',
        'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range',
        'fk_section_id'
    ]
    df = pd.DataFrame(rows, columns=columns)

    # Step 2: Calculate overall performance score
    performance_cols = [
        'avg_accuracy', 'avg_consistency', 'avg_speed',
        'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range'
    ]
    df['overall_performance_score'] = df[performance_cols].mean(axis=1)

    # Step 3: Create performance categories using quantiles
    q1 = df['overall_performance_score'].quantile(0.33)
    q2 = df['overall_performance_score'].quantile(0.67)

    bins = [df['overall_performance_score'].min() - 0.01, q1, q2, df['overall_performance_score'].max() + 0.01]
    labels = ['Low Performance', 'Medium Performance', 'High Performance']
    df['overall_performance_category'] = pd.cut(df['overall_performance_score'], bins=bins, labels=labels, right=True)

    # Step 4: Encode labels
    le = LabelEncoder()
    df['encoded_performance_category'] = le.fit_transform(df['overall_performance_category'])

    # Step 5: Train the Random Forest model
    X = df[performance_cols]
    y = df['encoded_performance_category']

    model = RandomForestClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=50,
        random_state=42
    )
    model.fit(X, y)

    # Step 6: Predict using the same dataset (no test/train split needed for demo)
    y_pred = model.predict(X)
    predicted_labels = le.inverse_transform(y_pred)

    df['Predicted Performance'] = predicted_labels

    # Step 7: Format response JSON
    response_data = df[['student_id', 'overall_performance_score', 'Predicted Performance', 'fk_section_id']].to_dict(orient='records')
    return JsonResponse({'predictions': response_data}, safe=False)