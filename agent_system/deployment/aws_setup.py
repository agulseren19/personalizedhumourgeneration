 #!/usr/bin/env python3
"""
AWS Cloud Setup for Humor Generation System
Handles infrastructure deployment and configuration
"""

import boto3
import json
import os
from typing import Dict, List, Any
import time

class AWSSetup:
    """Setup AWS infrastructure for humor generation system"""
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.session = boto3.Session(region_name=region)
        
        # Initialize clients
        self.dynamodb = self.session.client('dynamodb')
        self.iam = self.session.client('iam')
        self.lambda_client = self.session.client('lambda')
        self.apigateway = self.session.client('apigateway')
        self.ecs = self.session.client('ecs')
        self.ec2 = self.session.client('ec2')
        
    def setup_complete_infrastructure(self):
        """Setup complete AWS infrastructure"""
        print("🚀 Setting up AWS infrastructure for humor generation system...")
        
        # 1. Create IAM roles
        self.create_iam_roles()
        
        # 2. Setup DynamoDB tables
        self.setup_dynamodb_tables()
        
        # 3. Setup ECS cluster for Streamlit app
        self.setup_ecs_cluster()
        
        # 4. Setup API Gateway
        self.setup_api_gateway()
        
        # 5. Create Lambda functions
        self.create_lambda_functions()
        
        print("✅ AWS infrastructure setup complete!")
        self.print_deployment_info()
    
    def create_iam_roles(self):
        """Create necessary IAM roles"""
        print("📋 Creating IAM roles...")
        
        # Lambda execution role
        lambda_trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            self.iam.create_role(
                RoleName='HumorGenLambdaRole',
                AssumeRolePolicyDocument=json.dumps(lambda_trust_policy),
                Description='Role for humor generation Lambda functions'
            )
            
            # Attach policies
            self.iam.attach_role_policy(
                RoleName='HumorGenLambdaRole',
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Custom policy for DynamoDB and Bedrock
            custom_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:DeleteItem",
                            "dynamodb:Query",
                            "dynamodb:Scan"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:InvokeModel"
                        ],
                        "Resource": "*"
                    }
                ]
            }
            
            self.iam.put_role_policy(
                RoleName='HumorGenLambdaRole',
                PolicyName='HumorGenCustomPolicy',
                PolicyDocument=json.dumps(custom_policy)
            )
            
            print("✅ IAM roles created")
            
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️ IAM roles already exist")
            else:
                print(f"❌ Error creating IAM roles: {e}")
    
    def setup_dynamodb_tables(self):
        """Create DynamoDB tables"""
        print("🗄️ Creating DynamoDB tables...")
        
        tables = [
            {
                'TableName': 'humor-user-preferences',
                'KeySchema': [
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ]
            },
            {
                'TableName': 'humor-patterns',
                'KeySchema': [
                    {'AttributeName': 'pattern_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'pattern_id', 'AttributeType': 'S'},
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                'GlobalSecondaryIndexes': [
                    {
                        'IndexName': 'user-id-index',
                        'KeySchema': [
                            {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ]
            },
            {
                'TableName': 'humor-groups',
                'KeySchema': [
                    {'AttributeName': 'group_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'group_id', 'AttributeType': 'S'}
                ]
            }
        ]
        
        for table_config in tables:
            try:
                self.dynamodb.create_table(
                    TableName=table_config['TableName'],
                    KeySchema=table_config['KeySchema'],
                    AttributeDefinitions=table_config['AttributeDefinitions'],
                    BillingMode='PAY_PER_REQUEST',
                    **({k: v for k, v in table_config.items() 
                        if k not in ['TableName', 'KeySchema', 'AttributeDefinitions']})
                )
                
                print(f"✅ Created table: {table_config['TableName']}")
                
                # Wait for table to be active
                waiter = self.dynamodb.get_waiter('table_exists')
                waiter.wait(TableName=table_config['TableName'])
                
            except Exception as e:
                if "already exists" in str(e):
                    print(f"ℹ️ Table {table_config['TableName']} already exists")
                else:
                    print(f"❌ Error creating table {table_config['TableName']}: {e}")
    
    def setup_ecs_cluster(self):
        """Setup ECS cluster for Streamlit app"""
        print("🐳 Setting up ECS cluster...")
        
        try:
            # Create ECS cluster
            cluster_response = self.ecs.create_cluster(
                clusterName='humor-generation-cluster',
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            
            print("✅ ECS cluster created")
            
            # Create task definition
            task_definition = {
                "family": "humor-streamlit-app",
                "networkMode": "awsvpc",
                "requiresCompatibilities": ["FARGATE"],
                "cpu": "512",
                "memory": "1024",
                "executionRoleArn": f"arn:aws:iam::{self.get_account_id()}:role/HumorGenLambdaRole",
                "taskRoleArn": f"arn:aws:iam::{self.get_account_id()}:role/HumorGenLambdaRole",
                "containerDefinitions": [
                    {
                        "name": "streamlit-app",
                        "image": "humor-gen:latest", 
                        "portMappings": [
                            {
                                "containerPort": 8501,
                                "protocol": "tcp"
                            }
                        ],
                        "environment": [
                            {"name": "AWS_DEFAULT_REGION", "value": self.region},
                            {"name": "STREAMLIT_SERVER_PORT", "value": "8501"}
                        ],
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            "options": {
                                "awslogs-group": "/ecs/humor-streamlit",
                                "awslogs-region": self.region,
                                "awslogs-stream-prefix": "ecs"
                            }
                        }
                    }
                ]
            }
            
            self.ecs.register_task_definition(**task_definition)
            print("✅ ECS task definition created")
            
        except Exception as e:
            print(f"❌ Error setting up ECS: {e}")
    
    def create_lambda_functions(self):
        """Create Lambda functions for API endpoints"""
        print("⚡ Creating Lambda functions...")
        
        lambda_functions = [
            {
                'FunctionName': 'humor-generate',
                'Handler': 'lambda_function.lambda_handler',
                'Description': 'Generate humor using multi-agent system'
            },
            {
                'FunctionName': 'humor-feedback',
                'Handler': 'feedback_handler.lambda_handler',
                'Description': 'Handle user feedback and update preferences'
            },
            {
                'FunctionName': 'humor-personas',
                'Handler': 'persona_handler.lambda_handler',
                'Description': 'Manage personas and recommendations'
            }
        ]
        
        for func_config in lambda_functions:
            try:
                # Create deployment package (simplified)
                zip_content = self.create_lambda_zip(func_config['FunctionName'])
                
                self.lambda_client.create_function(
                    FunctionName=func_config['FunctionName'],
                    Runtime='python3.9',
                    Role=f"arn:aws:iam::{self.get_account_id()}:role/HumorGenLambdaRole",
                    Handler=func_config['Handler'],
                    Code={'ZipFile': zip_content},
                    Description=func_config['Description'],
                    Timeout=30,
                    MemorySize=512
                )
                
                print(f"✅ Created Lambda function: {func_config['FunctionName']}")
                
            except Exception as e:
                if "already exists" in str(e):
                    print(f"ℹ️ Lambda function {func_config['FunctionName']} already exists")
                else:
                    print(f"❌ Error creating Lambda function {func_config['FunctionName']}: {e}")
    
    def create_lambda_zip(self, function_name: str) -> bytes:
        """Create Lambda deployment package"""
        # Simplified lambda code
        lambda_code = f'''
import json
import boto3

def lambda_handler(event, context):
    """
    Lambda function for {function_name}
    """
    try:
        return {{
            'statusCode': 200,
            'headers': {{
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }},
            'body': json.dumps({{
                'message': 'Function {function_name} executed successfully',
                'event': event
            }})
        }}
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''
        
        # Simple zip creation (in practice, use proper packaging)
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('lambda_function.py', lambda_code)
        
        return zip_buffer.getvalue()
    
    def setup_api_gateway(self):
        """Setup API Gateway for HTTP endpoints"""
        print("🌐 Setting up API Gateway...")
        
        try:
            # Create REST API
            api_response = self.apigateway.create_rest_api(
                name='humor-generation-api',
                description='API for humor generation system',
                endpointConfiguration={'types': ['REGIONAL']}
            )
            
            api_id = api_response['id']
            print(f"✅ Created API Gateway: {api_id}")
            
            # Get root resource
            resources = self.apigateway.get_resources(restApiId=api_id)
            root_resource_id = resources['items'][0]['id']
            
            # Create resources and methods
            endpoints = [
                {'path': 'generate', 'method': 'POST', 'lambda': 'humor-generate'},
                {'path': 'feedback', 'method': 'POST', 'lambda': 'humor-feedback'},
                {'path': 'personas', 'method': 'GET', 'lambda': 'humor-personas'}
            ]
            
            for endpoint in endpoints:
                # Create resource
                resource_response = self.apigateway.create_resource(
                    restApiId=api_id,
                    parentId=root_resource_id,
                    pathPart=endpoint['path']
                )
                
                resource_id = resource_response['id']
                
                # Create method
                self.apigateway.put_method(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod=endpoint['method'],
                    authorizationType='NONE'
                )
                
                # Integration with Lambda
                lambda_arn = f"arn:aws:lambda:{self.region}:{self.get_account_id()}:function:{endpoint['lambda']}"
                
                self.apigateway.put_integration(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod=endpoint['method'],
                    type='AWS_PROXY',
                    integrationHttpMethod='POST',
                    uri=f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
                )
                
                print(f"✅ Created endpoint: /{endpoint['path']}")
            
            # Deploy API
            deployment = self.apigateway.create_deployment(
                restApiId=api_id,
                stageName='prod'
            )
            
            print(f"✅ API deployed to: https://{api_id}.execute-api.{self.region}.amazonaws.com/prod")
            
        except Exception as e:
            print(f"❌ Error setting up API Gateway: {e}")
    
    def get_account_id(self):
        """Get AWS account ID"""
        sts = self.session.client('sts')
        return sts.get_caller_identity()['Account']
    
    def print_deployment_info(self):
        """Print deployment information"""
        print("\n" + "="*50)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("="*50)
        print(f"Region: {self.region}")
        print(f"Account ID: {self.get_account_id()}")
        print("\n📋 Resources Created:")
        print("• DynamoDB Tables:")
        print("  - humor-user-preferences")
        print("  - humor-patterns") 
        print("  - humor-groups")
        print("• Lambda Functions:")
        print("  - humor-generate")
        print("  - humor-feedback")
        print("  - humor-personas")
        print("• ECS Cluster: humor-generation-cluster")
        print("• API Gateway: humor-generation-api")
        print("\n🔑 Next Steps:")
        print("1. Set up environment variables:")
        print("   export OPENAI_API_KEY=key")
        print("   export ANTHROPIC_API_KEY=key") 
        print("   export DEEPSEEK_API_KEY=key")
        print("2. Build and push Docker image for Streamlit app")
        print("3. Update Lambda functions with actual code")
        print("4. Configure domain and SSL certificates")
        print("="*50)

def main():
    """Main setup function"""
    print("🎭 AWS Setup for Humor Generation System")
    print("="*50)
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        print("✅ AWS credentials found")
    except:
        print("❌ No AWS credentials found. Please configure AWS CLI first.")
        return
    
    # Setup infrastructure
    setup = AWSSetup()
    setup.setup_complete_infrastructure()

if __name__ == "__main__":
    main()