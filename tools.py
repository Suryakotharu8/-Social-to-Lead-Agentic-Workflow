def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API call to capture a qualified lead.
    In production, this would POST to a CRM or backend service.
    """
    print(f"[LEAD CAPTURED] Name: {name} | Email: {email} | Platform: {platform}")
    return (
        f"✅ Lead captured successfully!\n"
        f"  Name: {name}\n"
        f"  Email: {email}\n"
        f"  Platform: {platform}\n\n"
        f"Our team will reach out to you shortly. Welcome to AutoStream! 🎬"
    )
