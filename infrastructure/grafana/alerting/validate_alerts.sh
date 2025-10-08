#!/bin/bash
# Grafana Alerting Configuration Validator
# Validates YAML syntax and configuration structure

set -e

echo "================================================"
echo "Grafana Unified Alerting Configuration Validator"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check YAML syntax
check_yaml() {
    local file=$1
    echo -n "Checking YAML syntax: $file ... "

    if command -v yamllint &> /dev/null; then
        if yamllint -d relaxed "$file" &> /dev/null; then
            echo -e "${GREEN}✓ PASS${NC}"
        else
            echo -e "${RED}✗ FAIL${NC}"
            yamllint -d relaxed "$file"
            ((ERRORS++))
        fi
    elif command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" &> /dev/null; then
            echo -e "${GREEN}✓ PASS${NC}"
        else
            echo -e "${RED}✗ FAIL${NC}"
            python3 -c "import yaml; yaml.safe_load(open('$file'))"
            ((ERRORS++))
        fi
    else
        echo -e "${YELLOW}⚠ SKIP (no YAML validator found)${NC}"
        ((WARNINGS++))
    fi
}

# Function to check alert rule structure
check_alert_rules() {
    local file=$1
    echo -n "Checking alert rules structure: $file ... "

    # Check required fields
    if grep -q "apiVersion: 1" "$file" && \
       grep -q "groups:" "$file" && \
       grep -q "interval:" "$file" && \
       grep -q "rules:" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL (missing required fields)${NC}"
        ((ERRORS++))
    fi
}

# Function to check UID uniqueness
check_uid_uniqueness() {
    echo -n "Checking alert UID uniqueness ... "

    # Extract only alert rule UIDs (at correct indentation level)
    local uids=$(grep -h "^      - uid:" rules/*.yml | awk '{print $3}' | sort)
    local unique_uids=$(echo "$uids" | uniq)

    if [ "$uids" = "$unique_uids" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        local count=$(echo "$uids" | wc -l | tr -d ' ')
        echo "  → $count unique alert UIDs verified"
    else
        echo -e "${RED}✗ FAIL (duplicate UIDs found)${NC}"
        comm -23 <(echo "$uids") <(echo "$unique_uids")
        ((ERRORS++))
    fi
}

# Function to count alerts
count_alerts() {
    local file=$1
    local count=$(grep -c "^      - uid:" "$file" || echo 0)
    echo "  → $count alert rules found"
}

# Function to check contact points
check_contact_points() {
    local file=$1
    echo -n "Checking contact points configuration: $file ... "

    if grep -q "contactPoints:" "$file" && \
       grep -q "name:" "$file" && \
       grep -q "receivers:" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}"

        # List contact points
        local points=$(grep "name:" "$file" | grep -v "orgId" | awk '{print $2}' | tr -d ',')
        echo "  → Contact points: $(echo $points | tr '\n' ', ' | sed 's/,$//')"
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
    fi
}

# Function to check notification policies
check_notification_policies() {
    local file=$1
    echo -n "Checking notification policies: $file ... "

    if grep -q "policies:" "$file" && \
       grep -q "receiver:" "$file" && \
       grep -q "group_by:" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}"

        # List routing rules
        local routes=$(grep -c "- receiver:" "$file" || echo 0)
        echo "  → $routes routing rules configured"
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ERRORS++))
    fi
}

# Function to validate Prometheus expressions
validate_prometheus_expr() {
    echo -n "Validating Prometheus expressions ... "

    local invalid_exprs=0
    for file in rules/*.yml; do
        # Check for common Prometheus expression issues
        if grep -q "expr: \"\"" "$file" || grep -q "expr:$" "$file"; then
            echo -e "${RED}✗ FAIL (empty expressions found in $file)${NC}"
            ((invalid_exprs++))
        fi
    done

    if [ $invalid_exprs -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        ((ERRORS++))
    fi
}

# Function to check dashboard links
check_dashboard_links() {
    echo -n "Checking dashboard UID references ... "

    local dashboard_uids=(
        "trading-pnl"
        "trading-execution"
        "trading-strategy"
        "trading-risk"
        "advanced-risk"
        "system-health"
    )

    local missing_refs=0
    for uid in "${dashboard_uids[@]}"; do
        if ! grep -q "dashboard_uid: $uid" rules/*.yml; then
            echo -e "\n  ${YELLOW}⚠ WARNING: No alerts linked to dashboard '$uid'${NC}"
            ((WARNINGS++))
        fi
    done

    echo -e "${GREEN}✓ Complete${NC}"
}

# Main validation
echo "1. Validating YAML Syntax"
echo "-------------------------"
check_yaml "alerting.yml"
check_yaml "contact-points.yml"
check_yaml "notification-policies.yml"
for file in rules/*.yml; do
    check_yaml "$file"
done
echo ""

echo "2. Validating Alert Rules Structure"
echo "-----------------------------------"
for file in rules/*.yml; do
    check_alert_rules "$file"
    count_alerts "$file"
done
echo ""

echo "3. Validating Alert UIDs"
echo "------------------------"
check_uid_uniqueness
echo ""

echo "4. Validating Contact Points"
echo "----------------------------"
check_contact_points "contact-points.yml"
echo ""

echo "5. Validating Notification Policies"
echo "-----------------------------------"
check_notification_policies "notification-policies.yml"
echo ""

echo "6. Validating Prometheus Expressions"
echo "------------------------------------"
validate_prometheus_expr
echo ""

echo "7. Validating Dashboard References"
echo "----------------------------------"
check_dashboard_links
echo ""

# Summary
echo "================================================"
echo "Validation Summary"
echo "================================================"
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All validation checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env and configure alert channels"
    echo "2. Run: docker-compose up -d grafana"
    echo "3. Access Grafana: http://localhost:3001"
    echo "4. Navigate to: Alerting → Alert rules"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS errors${NC}"
    echo "Please fix the errors above and run validation again."
    exit 1
fi
