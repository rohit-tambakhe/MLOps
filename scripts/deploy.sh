#!/bin/bash

# MLOps Pipeline Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPOSITORY=${ECR_REPOSITORY:-mlops-inference}
LAMBDA_FUNCTION=${LAMBDA_FUNCTION:-mlops-cifar-inference}
PROJECT_NAME=${PROJECT_NAME:-mlops-pipeline}

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install it first."
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        print_warning "Terraform not found. Infrastructure deployment will be skipped."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    print_header "Deploying infrastructure with Terraform..."
    
    if command -v terraform &> /dev/null; then
        cd infrastructure/terraform
        
        print_status "Initializing Terraform..."
        terraform init
        
        print_status "Planning infrastructure changes..."
        terraform plan -out=tfplan
        
        print_status "Applying infrastructure changes..."
        terraform apply tfplan
        
        # Get outputs
        ECR_URI=$(terraform output -raw ecr_repository_url)
        print_status "ECR Repository: $ECR_URI"
        
        cd ../..
    else
        print_warning "Terraform not found, skipping infrastructure deployment"
    fi
}

# Build and push Docker image
build_and_push_image() {
    print_header "Building and pushing Docker image..."
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY"
    
    print_status "Building Docker image..."
    docker build -f docker/Dockerfile.inference -t $ECR_REPOSITORY:latest .
    
    print_status "Logging into ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    print_status "Tagging image for ECR..."
    docker tag $ECR_REPOSITORY:latest $ECR_URI:latest
    docker tag $ECR_REPOSITORY:latest $ECR_URI:$(git rev-parse --short HEAD)
    
    print_status "Pushing image to ECR..."
    docker push $ECR_URI:latest
    docker push $ECR_URI:$(git rev-parse --short HEAD)
    
    print_status "Image pushed successfully ✓"
    echo "Image URI: $ECR_URI:latest"
}

# Update Lambda function
update_lambda() {
    print_header "Updating Lambda function..."
    
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY"
    IMAGE_URI="$ECR_URI:$(git rev-parse --short HEAD)"
    
    print_status "Updating Lambda function code..."
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION \
        --image-uri $IMAGE_URI \
        --region $AWS_REGION
    
    print_status "Waiting for function update to complete..."
    aws lambda wait function-updated \
        --function-name $LAMBDA_FUNCTION \
        --region $AWS_REGION
    
    print_status "Publishing new version..."
    VERSION=$(aws lambda publish-version \
        --function-name $LAMBDA_FUNCTION \
        --description "Deployed from commit $(git rev-parse --short HEAD)" \
        --region $AWS_REGION \
        --query Version --output text)
    
    print_status "Lambda function updated to version $VERSION ✓"
}

# Run integration tests
run_integration_tests() {
    print_header "Running integration tests..."
    
    # Get API Gateway URL
    API_URL=$(aws lambda get-function-url-config \
        --function-name $LAMBDA_FUNCTION \
        --region $AWS_REGION \
        --query FunctionUrl --output text 2>/dev/null || echo "")
    
    if [ -z "$API_URL" ]; then
        print_warning "Function URL not found, skipping integration tests"
        return
    fi
    
    print_status "Testing API endpoint: $API_URL"
    
    # Health check
    print_status "Running health check..."
    if curl -f -s "${API_URL}health" > /dev/null; then
        print_status "Health check passed ✓"
    else
        print_error "Health check failed"
        return 1
    fi
    
    print_status "Integration tests completed ✓"
}

# Cleanup old versions
cleanup_old_versions() {
    print_header "Cleaning up old versions..."
    
    print_status "Removing old Lambda versions..."
    aws lambda list-versions-by-function \
        --function-name $LAMBDA_FUNCTION \
        --region $AWS_REGION \
        --query 'Versions[?Version!=`$LATEST`].Version' \
        --output text | head -n -5 | xargs -I {} aws lambda delete-version \
        --function-name $LAMBDA_FUNCTION \
        --version {} \
        --region $AWS_REGION 2>/dev/null || true
    
    print_status "Cleanup completed ✓"
}

# Send notification
send_notification() {
    print_header "Sending deployment notification..."
    
    COMMIT_HASH=$(git rev-parse --short HEAD)
    COMMIT_MSG=$(git log -1 --pretty=%B)
    
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 MLOps Pipeline deployed successfully!\n\nCommit: \`$COMMIT_HASH\`\nMessage: $COMMIT_MSG\"}" \
            $SLACK_WEBHOOK_URL
        print_status "Slack notification sent ✓"
    else
        print_warning "SLACK_WEBHOOK_URL not set, skipping notification"
    fi
}

# Main deployment function
main() {
    echo "================================================"
    echo "       MLOps Pipeline Deployment"
    echo "================================================"
    echo
    echo "Region: $AWS_REGION"
    echo "Repository: $ECR_REPOSITORY"
    echo "Function: $LAMBDA_FUNCTION"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo
    
    check_prerequisites
    
    # Check if infrastructure should be deployed
    if [ "$1" = "--infra" ] || [ "$1" = "--full" ]; then
        deploy_infrastructure
    fi
    
    build_and_push_image
    update_lambda
    run_integration_tests
    cleanup_old_versions
    send_notification
    
    echo
    echo "================================================"
    print_status "Deployment completed successfully! 🎉"
    echo "================================================"
    echo
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --infra    Deploy infrastructure with Terraform"
    echo "  --full     Deploy infrastructure and application"
    echo "  --help     Show this help message"
    echo
    echo "Environment Variables:"
    echo "  AWS_REGION         AWS region (default: us-east-1)"
    echo "  ECR_REPOSITORY     ECR repository name (default: mlops-inference)"
    echo "  LAMBDA_FUNCTION    Lambda function name (default: mlops-cifar-inference)"
    echo "  SLACK_WEBHOOK_URL  Slack webhook for notifications"
    echo
}

# Parse command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
