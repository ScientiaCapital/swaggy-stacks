"""
GitHub webhook endpoints and automation API
"""

import hashlib
import hmac
from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.dependencies.mcp import get_github_service
from app.core.exceptions import MCPError
from app.services.github_automation import (
    GitHubAutomationService,
    PullRequestConfig,
    ReleaseConfig,
    WorkflowTrigger,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


class WebhookPayload(BaseModel):
    """GitHub webhook payload"""

    action: Optional[str] = None
    repository: Optional[Dict[str, Any]] = None
    pull_request: Optional[Dict[str, Any]] = None
    issue: Optional[Dict[str, Any]] = None
    ref: Optional[str] = None
    commits: Optional[list] = None


class PRCreateRequest(BaseModel):
    """Request model for PR creation"""

    title: str
    body: str
    head_branch: str
    base_branch: str = "main"
    draft: bool = False
    assignees: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class WorkflowTriggerRequest(BaseModel):
    """Request model for workflow triggers"""

    workflow_name: str
    branch: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    environment: Optional[str] = None


class ReleaseRequest(BaseModel):
    """Request model for release creation"""

    tag_name: str
    target_commitish: str = "main"
    name: Optional[str] = None
    body: Optional[str] = None
    draft: bool = False
    prerelease: bool = False


def verify_signature(
    payload_body: bytes, signature_header: str, webhook_secret: str
) -> bool:
    """Verify GitHub webhook signature"""
    if not signature_header:
        return False

    try:
        signature = signature_header.split("=")[1]
        expected_signature = hmac.new(
            webhook_secret.encode(), payload_body, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None),
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """Handle GitHub webhook events"""
    try:
        await request.body()

        # For production, you would verify the webhook signature
        # webhook_secret = "your_webhook_secret"
        # if not verify_signature(payload_body, x_hub_signature_256, webhook_secret):
        #     raise HTTPException(status_code=401, detail="Invalid signature")

        payload = await request.json()
        webhook_payload = WebhookPayload(**payload)

        # Process webhook in background
        background_tasks.add_task(
            process_github_webhook, x_github_event, webhook_payload, github_service
        )

        logger.info(
            "GitHub webhook received",
            event=x_github_event,
            action=webhook_payload.action,
        )

        return {"message": "Webhook processed successfully"}

    except Exception as e:
        logger.error(f"GitHub webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")


async def process_github_webhook(
    event_type: str, payload: WebhookPayload, github_service: GitHubAutomationService
):
    """Process GitHub webhook events in background"""
    try:
        if event_type == "pull_request":
            await handle_pull_request_event(payload, github_service)
        elif event_type == "push":
            await handle_push_event(payload, github_service)
        elif event_type == "issues":
            await handle_issue_event(payload, github_service)
        elif event_type == "release":
            await handle_release_event(payload, github_service)

        logger.info(f"Processed {event_type} webhook event successfully")

    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}")


async def handle_pull_request_event(
    payload: WebhookPayload, github_service: GitHubAutomationService
):
    """Handle pull request webhook events"""
    action = payload.action
    pr = payload.pull_request

    if not pr:
        return

    pr_number = pr.get("number")

    if action == "opened":
        # Add automated comment to new PRs
        welcome_comment = (
            "🚀 Thank you for your pull request!\n\n"
            "This PR will be automatically checked by our CI/CD pipeline. "
            "Please ensure all tests pass and code quality checks are satisfied.\n\n"
            "The automated deployment process will be triggered once this PR is merged to main."
        )

        await github_service.create_issue_comment(pr_number, welcome_comment)

    elif action == "closed" and pr.get("merged"):
        # Trigger deployment workflow if merged to main
        if pr.get("base", {}).get("ref") == "main":
            workflow_trigger = WorkflowTrigger(
                workflow_name="deploy-production",
                branch="main",
                environment="production",
            )
            await github_service.manage_deployment_workflow(workflow_trigger)


async def handle_push_event(
    payload: WebhookPayload, github_service: GitHubAutomationService
):
    """Handle push webhook events"""
    ref = payload.ref
    payload.commits or []

    # Trigger workflows based on branch
    if ref == "refs/heads/main":
        # Production deployment
        workflow_trigger = WorkflowTrigger(
            workflow_name="ci-cd-pipeline", branch="main", environment="production"
        )
        await github_service.manage_deployment_workflow(workflow_trigger)

    elif ref == "refs/heads/develop":
        # Staging deployment
        workflow_trigger = WorkflowTrigger(
            workflow_name="ci-cd-pipeline", branch="develop", environment="staging"
        )
        await github_service.manage_deployment_workflow(workflow_trigger)


async def handle_issue_event(
    payload: WebhookPayload, github_service: GitHubAutomationService
):
    """Handle issue webhook events"""
    action = payload.action
    issue = payload.issue

    if not issue:
        return

    if action == "opened":
        # Check if it's a deployment blocker
        labels = [label["name"] for label in issue.get("labels", [])]
        if "deployment-blocker" in labels:
            logger.warning(
                "Deployment blocker issue created",
                issue_number=issue["number"],
                title=issue["title"],
            )


async def handle_release_event(
    payload: WebhookPayload, github_service: GitHubAutomationService
):
    """Handle release webhook events"""
    action = payload.action

    if action == "published":
        logger.info(
            "Release published",
            tag_name=payload.repository.get("tag_name") if payload.repository else None,
        )


@router.post("/automation/create-pr")
async def create_pull_request(
    request: PRCreateRequest,
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """Create automated pull request"""
    try:
        config = PullRequestConfig(
            title=request.title,
            body=request.body,
            head_branch=request.head_branch,
            base_branch=request.base_branch,
            draft=request.draft,
            assignees=request.assignees,
            labels=request.labels,
        )

        result = await github_service.create_automated_pr(config)
        return result

    except MCPError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"PR creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PR creation failed: {e}")


@router.post("/automation/trigger-workflow")
async def trigger_deployment_workflow(
    request: WorkflowTriggerRequest,
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """Trigger deployment workflow"""
    try:
        trigger = WorkflowTrigger(
            workflow_name=request.workflow_name,
            branch=request.branch,
            inputs=request.inputs,
            environment=request.environment,
        )

        result = await github_service.manage_deployment_workflow(trigger)
        return result

    except MCPError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow trigger failed: {e}")


@router.post("/automation/create-release")
async def create_release(
    request: ReleaseRequest,
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """Create coordinated release"""
    try:
        config = ReleaseConfig(
            tag_name=request.tag_name,
            target_commitish=request.target_commitish,
            name=request.name,
            body=request.body,
            draft=request.draft,
            prerelease=request.prerelease,
        )

        result = await github_service.coordinate_releases(config)
        return result

    except MCPError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Release creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Release creation failed: {e}")


@router.get("/automation/workflow-status/{workflow_name}")
async def get_workflow_status(
    workflow_name: str,
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """Get GitHub workflow status"""
    try:
        result = await github_service.get_workflow_status(workflow_name)
        return result

    except MCPError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


@router.get("/automation/health")
async def github_automation_health(
    github_service: GitHubAutomationService = Depends(get_github_service),
):
    """GitHub automation service health check"""
    try:
        health_status = await github_service.health_check()

        status_code = 200
        if health_status["status"] == "unhealthy":
            status_code = 503
        elif health_status["status"] == "degraded":
            status_code = 200  # Still functional but with warnings

        return JSONResponse(content=health_status, status_code=status_code)

    except Exception as e:
        logger.error(f"GitHub health check failed: {e}")
        return JSONResponse(
            content={
                "service": "github_automation",
                "status": "unhealthy",
                "error": str(e),
            },
            status_code=503,
        )
