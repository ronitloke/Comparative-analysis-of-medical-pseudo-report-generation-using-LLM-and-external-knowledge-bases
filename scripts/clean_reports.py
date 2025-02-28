import re

def extract_report_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    reports = content.split("--------------------------------------------------------------------------------")
    cleaned_reports = {}

    for report in reports:
        image_id_match = re.search(r'Image ID: (\S+)', report)
        if image_id_match:
            image_id = image_id_match.group(1)
            
            report_content_match = re.search(r'content=\'(.*?)\'', report, re.DOTALL)
            if report_content_match:
                report_content = report_content_match.group(1).replace('\\n', ' ').replace('\\', '')

                cleaned_reports[image_id] = report_content.strip()
    
    return cleaned_reports
