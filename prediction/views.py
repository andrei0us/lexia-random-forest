# views.py
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from django.db import connection


@csrf_exempt
def predict_performance(request):
    # Load data from student_cluster_data
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT student_id,
                   avg_accuracy,
                   avg_consistency,
                   avg_speed,
                   avg_retention,
                   avg_problem_solving_skills,
                   avg_vocabulary_range,
                   fk_section_id
            FROM student_cluster_data
            WHERE avg_accuracy IS NOT NULL
              AND avg_consistency IS NOT NULL
              AND avg_speed IS NOT NULL
              AND avg_retention IS NOT NULL
              AND avg_problem_solving_skills IS NOT NULL
              AND avg_vocabulary_range IS NOT NULL
        """)
        rows = cursor.fetchall()

    # Column labels
    columns = [
        'student_id', 'avg_accuracy', 'avg_consistency', 'avg_speed',
        'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range',
        'fk_section_id'
    ]
    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return JsonResponse({'error': 'No valid data found for prediction.'}, status=400)

    # Calculate average performance
    performance_cols = [
        'avg_accuracy', 'avg_consistency', 'avg_speed',
        'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range'
    ]
    df['overall_performance_score'] = df[performance_cols].mean(axis=1)

    # Define bins and categorize
    q1 = df['overall_performance_score'].quantile(0.33)
    q2 = df['overall_performance_score'].quantile(0.67)
    bins = [df['overall_performance_score'].min() - 0.01, q1, q2, df['overall_performance_score'].max() + 0.01]
    labels = ['Low Performance', 'Medium Performance', 'High Performance']
    df['overall_performance_category'] = pd.cut(df['overall_performance_score'], bins=bins, labels=labels, right=True)

    # Encode for model training
    le = LabelEncoder()
    df['encoded_performance_category'] = le.fit_transform(df['overall_performance_category'])

    # Train Random Forest
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

    # Predict
    y_pred = model.predict(X)
    df['Predicted Performance'] = le.inverse_transform(y_pred)

    # Save back to DB
    with connection.cursor() as cursor:
        for _, row in df.iterrows():
            cursor.execute("""
                UPDATE student_cluster_data
                SET pred_performance = %s
                WHERE student_id = %s
            """, [row['Predicted Performance'], row['student_id']])

    # Return JSON
    response_data = df[[
        'student_id', 'overall_performance_score', 'Predicted Performance', 'fk_section_id'
    ]].to_dict(orient='records')

    return JsonResponse({'success': True, 'predictions': response_data}, status=200)
