"""
Example: EU AI Act Conformity Assessment

Demonstrates how to perform automated conformity assessment for EU AI Act compliance.
This example shows a complete workflow:
1. Create a monitored LangChain system with all compliance features enabled
2. Capture system metadata
3. Perform risk assessment
4. Conduct conformity assessment
5. Generate compliance reports
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from aiact_toolkit.langchain_monitor import LangChainMonitor
from aiact_toolkit.risk_assessment import AIActRiskAssessor
from aiact_toolkit.conformity_assessment import ConformityAssessor, generate_conformity_report

# Initialize monitor with all compliance features enabled
monitor = LangChainMonitor(
    system_name="Medical Advisory Chatbot",
    enable_metrics=True,
    enable_audit_trail=True,
    enable_versioning=True,
    enable_data_governance=True
)

# Start monitoring
monitor.start()

# Create a medical advisory chatbot
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    callbacks=[monitor.callback_handler]
)

prompt = PromptTemplate(
    input_variables=["symptom"],
    template="You are a medical advisor. A patient reports: {symptom}\n\nProvide general health advice."
)

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[monitor.callback_handler])

# Simulate some operations
print("Running medical chatbot operations...")
try:
    response1 = chain.run(symptom="headache and fever")
    print(f"Response 1: {response1[:100]}...")

    response2 = chain.run(symptom="persistent cough")
    print(f"Response 2: {response2[:100]}...")
except Exception as e:
    print(f"Note: Operations failed (this is expected in demo): {e}")
    print("Continuing with conformity assessment...\n")

# Get metadata
metadata = monitor.get_metadata()

# Add data governance information
print("Adding data governance information...")
data_tracker = monitor.storage.get_data_governance_tracker()
if data_tracker:
    # Add training data source
    data_tracker.add_data_source(
        source_id="medical_training_data",
        name="Medical Advisory Training Dataset",
        data_type="text",
        description="Curated medical Q&A dataset for training advisory models",
        location="internal_database",
        size_records=50000,
        quality_metrics={
            "completeness": {"value": 0.95, "checked_at": "2024-01-15"},
            "accuracy": {"value": 0.92, "checked_at": "2024-01-15"}
        },
        personal_data=False,
        sensitive_data=True,
        license_info="Internal use only",
        provenance="Licensed from medical institution"
    )

# Perform risk assessment
print("\n" + "="*80)
print("STEP 1: RISK ASSESSMENT")
print("="*80 + "\n")

risk_assessor = AIActRiskAssessor()
risk_assessment = risk_assessor.assess_risk(
    metadata=metadata,
    use_case="Medical advisory chatbot providing general health advice based on symptoms",
    application_domain="healthcare"
)

print(f"Risk Level: {risk_assessment['risk_level'].upper()}")
print(f"Confidence: {risk_assessment['confidence'] * 100}%")
print("\nRisk Factors:")
for factor in risk_assessment['risk_factors']:
    print(f"  • {factor}")

# Save risk assessment to metadata
monitor.storage.set_risk_assessment(risk_assessment)
metadata = monitor.get_metadata()

# Perform conformity assessment
print("\n" + "="*80)
print("STEP 2: CONFORMITY ASSESSMENT")
print("="*80 + "\n")

conformity_assessor = ConformityAssessor()
conformity_result = conformity_assessor.assess_compliance(metadata)

# Print summary report
summary = generate_conformity_report(conformity_result)
print(summary)

# Show detailed findings for critical gaps
if conformity_result.critical_gaps:
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF CRITICAL GAPS")
    print("="*80 + "\n")

    for i, gap in enumerate(conformity_result.critical_gaps, 1):
        print(f"{i}. {gap}\n")

# Show actionable recommendations
if conformity_result.recommendations:
    print("\n" + "="*80)
    print("ACTIONABLE RECOMMENDATIONS")
    print("="*80 + "\n")

    for i, rec in enumerate(conformity_result.recommendations, 1):
        print(f"{i}. {rec}\n")

# Show compliance improvement pathway
print("\n" + "="*80)
print("COMPLIANCE IMPROVEMENT PATHWAY")
print("="*80 + "\n")

compliance_rate = (conformity_result.requirements_passed / conformity_result.requirements_checked * 100) if conformity_result.requirements_checked > 0 else 0

if conformity_result.overall_status.value == "compliant":
    print("✓ SYSTEM IS COMPLIANT")
    print("\nYour system meets all applicable EU AI Act requirements.")
    print("\nNext steps:")
    print("  1. Maintain compliance through regular monitoring")
    print("  2. Update documentation when system changes")
    print("  3. Implement post-market monitoring (Article 72)")
elif conformity_result.overall_status.value == "partial":
    print(f"~ PARTIAL COMPLIANCE ({compliance_rate:.0f}%)")
    print("\nYour system meets core requirements but has documentation gaps.")
    print("\nPriority actions:")

    # Identify quick wins
    partial_reqs = [r for r in conformity_result.detailed_results
                   if r.status.value == "partial"]
    failed_reqs = [r for r in conformity_result.detailed_results
                  if r.status.value == "non_compliant"]

    if partial_reqs:
        print(f"  1. Complete {len(partial_reqs)} partially fulfilled requirements")
        print("     (These require documentation completion, not new implementations)")

    if failed_reqs:
        mandatory_failed = [r for r in failed_reqs if r.mandatory]
        optional_failed = [r for r in failed_reqs if not r.mandatory]

        if mandatory_failed:
            print(f"  2. Implement {len(mandatory_failed)} missing mandatory controls")
        if optional_failed:
            print(f"  3. Consider implementing {len(optional_failed)} optional requirements")

    print("\n  After improvements, re-run: aiact-toolkit conformity-assessment metadata.json")
else:
    print(f"✗ NON-COMPLIANT ({compliance_rate:.0f}%)")
    print("\nYour system has critical compliance gaps that must be addressed.")
    print("\nIMPORTANT: Do not deploy this system until compliance is achieved.")
    print("\nCritical actions required:")

    critical_failed = [r for r in conformity_result.detailed_results
                      if r.mandatory and r.status.value == "non_compliant"]

    for req in critical_failed[:3]:  # Show top 3 critical issues
        print(f"\n  • {req.article}: {req.description}")
        if req.findings:
            print(f"    Issue: {req.findings[0]}")

# Save metadata with assessment results
output_file = "medical_chatbot_metadata.json"
monitor.save_to_file(output_file)

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80 + "\n")
print(f"✓ Metadata: {output_file}")
print(f"✓ Audit Trail: {output_file.replace('.json', '.audit.json')}")
print(f"✓ Version History: {output_file.replace('.json', '.versions.json')}")
print(f"✓ Data Governance: {output_file.replace('.json', '.data_governance.json')}")

print("\nTo generate detailed compliance reports, run:")
print(f"  aiact-toolkit conformity-assessment {output_file} -o conformity_report.md")
print(f"  aiact-toolkit generate-technical-doc {output_file} -o technical_doc.md")
print(f"  aiact-toolkit generate-model-card {output_file} -o model_card.md")

print("\n" + "="*80)
print("CONFORMITY ASSESSMENT COMPLETE")
print("="*80)
