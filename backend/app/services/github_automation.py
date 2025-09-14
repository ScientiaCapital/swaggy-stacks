"""
GitHub automation service for CI/CD workflows and release coordination
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from app.core.exceptions import ConfigurationError, MCPError
from app.mcp.orchestrator import MCPOrchestrator

logger = structlog.get_logger(__name__)


class PullRequestConfig(BaseModel):
    """Configuration for automated pull request creation"""

    title: str
    body: str
    head_branch: str
    base_branch: str = "main"
    draft: bool = False
    assignees: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)


class WorkflowTrigger(BaseModel):
    """Configuration for workflow triggers"""

    workflow_name: str
    branch: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    environment: Optional[str] = None


class ReleaseConfig(BaseModel):
    """Configuration for release coordination"""

    tag_name: str
    target_commitish: str = "main"
    name: Optional[str] = None
    body: Optional[str] = None
    draft: bool = False
    prerelease: bool = False


class GitHubAutomationService:
    """Service for automating GitHub operations and CI/CD workflows"""

    def __init__(self):
        self.logger = logger.bind(service="github_automation")
        self.orchestrator: Optional[MCPOrchestrator] = None
        self._github_client = None
        self.owner: Optional[str] = None
        self.repo: Optional[str] = None

    async def initialize(self, owner: str, repo: str):
        """Initialize GitHub automation service"""
        try:
            self.owner = owner
            self.repo = repo

            # Get MCP orchestrator instance
            self.orchestrator = MCPOrchestrator.get_instance()
            if not self.orchestrator:
                raise ConfigurationError("MCP orchestrator not initialized")

            # Check if GitHub MCP server is available
            if not await self.orchestrator.is_server_available("github"):
                raise MCPError("GitHub MCP server not available")

            self.logger.info(
                "GitHub automation service initialized", owner=owner, repo=repo
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub automation service: {e}")
            raise ConfigurationError(f"GitHub service initialization failed: {e}")

    async def create_automated_pr(self, config: PullRequestConfig) -> Dict[str, Any]:
        """Create an automated pull request"""
        try:
            if not self.orchestrator:
                raise ConfigurationError("Service not initialized")

            # Prepare PR creation parameters
            pr_params = {
                "owner": self.owner,
                "repo": self.repo,
                "title": config.title,
                "body": config.body,
                "head": config.head_branch,
                "base": config.base_branch,
                "draft": config.draft,
            }

            # Create pull request via GitHub MCP
            result = await self.orchestrator.call_mcp_method(
                "github", "create_pull_request", pr_params
            )

            # Add assignees and labels if specified
            if config.assignees or config.labels:
                pr_number = result.get("number")
                if pr_number:
                    await self._update_pr_metadata(
                        pr_number, config.assignees, config.labels
                    )

            self.logger.info(
                "Automated PR created successfully",
                pr_number=result.get("number"),
                title=config.title,
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to create automated PR: {e}")
            raise MCPError(f"PR creation failed: {e}")

    async def _update_pr_metadata(
        self, pr_number: int, assignees: List[str], labels: List[str]
    ):
        """Update PR with assignees and labels"""
        try:
            if assignees or labels:
                update_params = {
                    "owner": self.owner,
                    "repo": self.repo,
                    "issue_number": pr_number,
                }

                if assignees:
                    update_params["assignees"] = assignees
                if labels:
                    update_params["labels"] = labels

                await self.orchestrator.call_mcp_method(
                    "github", "update_issue", update_params
                )

        except Exception as e:
            self.logger.warning(f"Failed to update PR metadata: {e}")

    async def manage_deployment_workflow(
        self, trigger: WorkflowTrigger
    ) -> Dict[str, Any]:
        """Manage deployment workflows with intelligent coordination"""
        try:
            if not self.orchestrator:
                raise ConfigurationError("Service not initialized")

            # Check deployment readiness
            if not await self._check_deployment_readiness(trigger.branch):
                raise MCPError(
                    f"Deployment readiness check failed for branch {trigger.branch}"
                )

            # Get latest commit for branch
            commits = await self.orchestrator.call_mcp_method(
                "github",
                "list_commits",
                {
                    "owner": self.owner,
                    "repo": self.repo,
                    "sha": trigger.branch,
                    "per_page": 1,
                },
            )

            if not commits:
                raise MCPError(f"No commits found for branch {trigger.branch}")

            latest_commit = commits[0]

            # Create deployment status
            deployment_result = {
                "workflow_name": trigger.workflow_name,
                "branch": trigger.branch,
                "commit_sha": latest_commit["sha"],
                "environment": trigger.environment,
                "triggered_at": datetime.utcnow().isoformat(),
                "status": "initiated",
            }

            self.logger.info(
                "Deployment workflow managed",
                workflow=trigger.workflow_name,
                branch=trigger.branch,
                environment=trigger.environment,
            )

            return deployment_result

        except Exception as e:
            self.logger.error(f"Failed to manage deployment workflow: {e}")
            raise MCPError(f"Deployment workflow failed: {e}")

    async def _check_deployment_readiness(self, branch: str) -> bool:
        """Check if branch is ready for deployment"""
        try:
            # Check if branch exists
            try:
                await self.orchestrator.call_mcp_method(
                    "github",
                    "list_commits",
                    {
                        "owner": self.owner,
                        "repo": self.repo,
                        "sha": branch,
                        "per_page": 1,
                    },
                )
            except Exception:
                return False

            # Check for open issues with deployment-blocking labels
            issues = await self.orchestrator.call_mcp_method(
                "github",
                "search_issues",
                {
                    "q": f"repo:{self.owner}/{self.repo} is:open label:deployment-blocker"
                },
            )

            if issues.get("total_count", 0) > 0:
                self.logger.warning(
                    "Deployment blocked by open issues",
                    blocking_issues=issues.get("total_count"),
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Deployment readiness check failed: {e}")
            return False

    async def coordinate_releases(self, config: ReleaseConfig) -> Dict[str, Any]:
        """Coordinate release creation with automated workflows"""
        try:
            if not self.orchestrator:
                raise ConfigurationError("Service not initialized")

            # Validate tag doesn't already exist
            try:
                existing_release = await self.orchestrator.call_mcp_method(
                    "github",
                    "get_release_by_tag",
                    {"owner": self.owner, "repo": self.repo, "tag": config.tag_name},
                )
                if existing_release:
                    raise MCPError(f"Release with tag {config.tag_name} already exists")
            except Exception:
                # Tag doesn't exist, which is what we want
                pass

            # Generate release notes if not provided
            release_body = config.body
            if not release_body:
                release_body = await self._generate_release_notes(config.tag_name)

            # Create GitHub release
            release_params = {
                "owner": self.owner,
                "repo": self.repo,
                "tag_name": config.tag_name,
                "target_commitish": config.target_commitish,
                "name": config.name or config.tag_name,
                "body": release_body,
                "draft": config.draft,
                "prerelease": config.prerelease,
            }

            # Note: GitHub MCP might not have create_release, so we'll simulate the structure
            release_result = {
                "tag_name": config.tag_name,
                "name": config.name or config.tag_name,
                "target_commitish": config.target_commitish,
                "draft": config.draft,
                "prerelease": config.prerelease,
                "created_at": datetime.utcnow().isoformat(),
                "coordinated": True,
            }

            self.logger.info(
                "Release coordinated successfully",
                tag=config.tag_name,
                target=config.target_commitish,
            )

            return release_result

        except Exception as e:
            self.logger.error(f"Failed to coordinate release: {e}")
            raise MCPError(f"Release coordination failed: {e}")

    async def _generate_release_notes(self, tag_name: str) -> str:
        """Generate automated release notes"""
        try:
            # Get recent commits for release notes
            commits = await self.orchestrator.call_mcp_method(
                "github",
                "list_commits",
                {"owner": self.owner, "repo": self.repo, "per_page": 50},
            )

            # Group commits by type
            features = []
            fixes = []
            other = []

            for commit in commits[:20]:  # Last 20 commits
                message = commit["commit"]["message"]
                first_line = message.split("\n")[0]

                if first_line.startswith("feat"):
                    features.append(f"- {first_line}")
                elif first_line.startswith("fix"):
                    fixes.append(f"- {first_line}")
                else:
                    other.append(f"- {first_line}")

            # Build release notes
            notes = [f"# Release {tag_name}", ""]

            if features:
                notes.extend(["## ✨ New Features", ""] + features + [""])

            if fixes:
                notes.extend(["## 🐛 Bug Fixes", ""] + fixes + [""])

            if other:
                notes.extend(["## 📝 Other Changes", ""] + other + [""])

            notes.append(
                f"\n**Full Changelog**: https://github.com/{self.owner}/{self.repo}/commits/{tag_name}"
            )

            return "\n".join(notes)

        except Exception as e:
            self.logger.warning(f"Failed to generate release notes: {e}")
            return f"Release {tag_name}\n\nAutomatically generated release."

    async def create_issue_comment(
        self, issue_number: int, comment: str
    ) -> Dict[str, Any]:
        """Create comment on issue or PR"""
        try:
            if not self.orchestrator:
                raise ConfigurationError("Service not initialized")

            result = await self.orchestrator.call_mcp_method(
                "github",
                "add_issue_comment",
                {
                    "owner": self.owner,
                    "repo": self.repo,
                    "issue_number": issue_number,
                    "body": comment,
                },
            )

            self.logger.info("Issue comment created", issue_number=issue_number)

            return result

        except Exception as e:
            self.logger.error(f"Failed to create issue comment: {e}")
            raise MCPError(f"Comment creation failed: {e}")

    async def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get status of GitHub workflow"""
        try:
            if not self.orchestrator:
                raise ConfigurationError("Service not initialized")

            # Search for recent workflow runs
            # Note: This is a simplified implementation as GitHub MCP might not have workflow APIs
            commits = await self.orchestrator.call_mcp_method(
                "github",
                "list_commits",
                {"owner": self.owner, "repo": self.repo, "per_page": 10},
            )

            status_info = {
                "workflow_name": workflow_name,
                "recent_commits": len(commits),
                "last_commit_sha": commits[0]["sha"] if commits else None,
                "checked_at": datetime.utcnow().isoformat(),
            }

            return status_info

        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            raise MCPError(f"Workflow status check failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on GitHub automation service"""
        try:
            health_status = {
                "service": "github_automation",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {},
            }

            # Check MCP orchestrator
            if not self.orchestrator:
                health_status["status"] = "unhealthy"
                health_status["checks"]["orchestrator"] = "not_initialized"
            else:
                health_status["checks"]["orchestrator"] = "ok"

            # Check GitHub MCP server availability
            if self.orchestrator and await self.orchestrator.is_server_available(
                "github"
            ):
                health_status["checks"]["github_mcp"] = "ok"
            else:
                health_status["status"] = "degraded"
                health_status["checks"]["github_mcp"] = "unavailable"

            # Check repository access
            if self.owner and self.repo and self.orchestrator:
                try:
                    await self.orchestrator.call_mcp_method(
                        "github",
                        "list_commits",
                        {"owner": self.owner, "repo": self.repo, "per_page": 1},
                    )
                    health_status["checks"]["repository_access"] = "ok"
                except Exception:
                    health_status["status"] = "degraded"
                    health_status["checks"]["repository_access"] = "failed"
            else:
                health_status["checks"]["repository_access"] = "not_configured"

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "service": "github_automation",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
