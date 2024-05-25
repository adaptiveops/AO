import boto3
import os

# Print AWS credentials to verify
print("AWS_ACCESS_KEY_ID:", os.getenv('AWS_ACCESS_KEY_ID'))
print("AWS_SECRET_ACCESS_KEY:", os.getenv('AWS_SECRET_ACCESS_KEY'))

# Try initializing a Polly client
try:
    polly_client = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    ).client('polly')

    response = polly_client.synthesize_speech(
        VoiceId='Joanna',
        OutputFormat='mp3',
        Text='Testing AWS Polly service.'
    )

    print("Polly service initialized successfully.")
except Exception as e:
    print(f"Error initializing Polly service: {e}")
