#!/usr/bin/env python3
"""
Test script to verify AWS S3 cloud connection
Run this after setting up your .env file to ensure cloud upload will work.
"""

import os
import sys
from pathlib import Path

def main() -> int:
    print("=" * 70)
    print("🔍 ECG MONITOR - CLOUD CONNECTION TEST")
    print("=" * 70)
    print()

# Step 1: Check if .env file exists
    print("Step 1: Checking for .env file...")
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ ERROR: .env file not found!")
        print()
        print("📝 To fix this:")
        print("   1. Copy the template: cp env_template.txt .env")
        print("   2. Edit .env with your AWS credentials")
        print("   3. Run this test again")
        print()
        return 1
    else:
        print("✅ .env file found")
    print()

# Step 2: Load environment variables
    print("Step 2: Loading environment variables...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("❌ ERROR: python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return 1
    print()

# Step 3: Check required environment variables
    print("Step 3: Checking AWS credentials...")
    required_vars = {
        'CLOUD_SERVICE': os.getenv('CLOUD_SERVICE'),
        'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
        'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'AWS_S3_BUCKET': os.getenv('AWS_S3_BUCKET'),
        'AWS_S3_REGION': os.getenv('AWS_S3_REGION')
    }

    missing = []
    for key, value in required_vars.items():
        if not value or value.startswith('your_'):
            print(f"❌ {key}: NOT SET or using placeholder")
            missing.append(key)
        else:
            # Mask sensitive values
            if 'SECRET' in key or 'ACCESS_KEY' in key:
                display_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {key}: {display_value}")

    if missing:
        print()
        print(f"❌ Missing or placeholder values: {', '.join(missing)}")
        print()
        print("📝 To fix this:")
        print("   1. Open .env file: nano .env")
        print("   2. Replace placeholder values with real AWS credentials")
        print("   3. Ask Divyansh for the correct credentials")
        print()
        return 1
    print()

# Step 4: Test boto3 import
    print("Step 4: Checking boto3 installation...")
    try:
        import boto3
        print("✅ boto3 is installed")
    except ImportError:
        print("❌ ERROR: boto3 not installed")
        print("   Run: pip install boto3")
        return 1
    print()

# Step 5: Test AWS connection
    print("Step 5: Testing AWS S3 connection...")
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION')
        )

        # Try to list objects in the bucket (this verifies credentials and bucket access)
        bucket_name = os.getenv('AWS_S3_BUCKET')
        s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)

        print(f"✅ Successfully connected to S3 bucket: {bucket_name}")
        print(f"✅ Region: {os.getenv('AWS_S3_REGION')}")

    except Exception as e:
        print(f"❌ ERROR: Failed to connect to AWS S3")
        print(f"   Error details: {str(e)}")
        print()
        print("📝 Common issues:")
        print("   - Invalid AWS credentials (check ACCESS_KEY_ID and SECRET_ACCESS_KEY)")
        print("   - Incorrect bucket name (check AWS_S3_BUCKET)")
        print("   - Incorrect region (check AWS_S3_REGION)")
        print("   - IAM user doesn't have S3 permissions")
        print()
        print("   Contact Divyansh to verify your credentials")
        return 1
    print()

# Step 6: Test upload capability (dry run)
    print("Step 6: Testing upload permissions...")
    try:
        # Create a tiny test file
        test_content = b"ECG Monitor Test"
        test_key = "test_connection.txt"

        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content
        )
        print(f"✅ Upload test successful")

        # Clean up test file
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"✅ Delete test successful")

    except Exception as e:
        print(f"❌ ERROR: Upload/delete test failed")
        print(f"   Error details: {str(e)}")
        print()
        print("📝 Your IAM user may not have write permissions")
        print("   Contact Divyansh to grant you S3 write access")
        return 1
    print()

# Success!
    print("=" * 70)
    print("🎉 SUCCESS! Cloud upload is properly configured!")
    print("=" * 70)
    print()
    print("✅ All checks passed:")
    print("   • .env file exists")
    print("   • All credentials are set")
    print("   • boto3 is installed")
    print("   • AWS connection works")
    print("   • Upload permissions verified")
    print()
    print("🚀 You're ready to use cloud sync in the ECG Monitor!")
    print()
    print("💡 Next steps:")
    print("   1. Run the app: python src/main.py")
    print("   2. Generate a report")
    print("   3. Click 'Cloud Sync' button on the dashboard")
    print("   4. Reports will automatically upload to S3")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
