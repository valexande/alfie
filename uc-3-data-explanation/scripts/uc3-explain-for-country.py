import pandas as pd
import requests


def load_data(file_path):
    """Load the Excel file and return data from both sheets."""
    df_a = pd.read_excel(file_path, sheet_name='a-part-of-data')  # Sheet with country data
    df_b = pd.read_excel(file_path, sheet_name='b-part-of-data', header=None)  # Sheet with Perceivable, Operable, etc.
    df_c = pd.read_excel(file_path, sheet_name='c-part-of-data', header=None)  # Sheet with detailed metrics
    return df_a, df_b, df_c


def check_website_status(url):
    """Check if the website is accessible by getting a 20X response."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code >= 200 and response.status_code < 300:
            return "Website is accessible."
        else:
            return "Website is not accessible."
    except requests.RequestException:
        return "Website is not accessible."


def evaluate_accessibility(country, df_a):
    """Evaluate accessibility for a given country."""
    country_data = df_a[df_a['Country'].str.lower() == country.lower()]
    if country_data.empty:
        return f"No data available for {country}."

    row = country_data.iloc[0]
    url = row['URL']
    institution = row['Institution']
    domain = row['Domain']

    # Convert relevant fields to numeric values, forcing errors to NaN
    total_errors = pd.to_numeric(row['Errors'], errors='coerce')
    contrast_issues = pd.to_numeric(row['Contrast Errors'], errors='coerce')
    alerts = pd.to_numeric(row['Alerts'], errors='coerce')
    missing_alt_text = pd.to_numeric(row['Missing alternative text'], errors='coerce')
    keyboard_issues = pd.to_numeric(row['Keyboard (Level A)'], errors='coerce')
    headings_labels = pd.to_numeric(row['Headings and Labels (Level AA)'], errors='coerce')
    observation_labels = row['Observation'] if isinstance(row['Observation'], str) else "No specific observations."

    # Handle missing values (NaN) by replacing them with 0
    total_errors = 0 if pd.isna(total_errors) else total_errors
    contrast_issues = 0 if pd.isna(contrast_issues) else contrast_issues
    alerts = 0 if pd.isna(alerts) else alerts
    missing_alt_text = 0 if pd.isna(missing_alt_text) else missing_alt_text
    keyboard_issues = 0 if pd.isna(keyboard_issues) else keyboard_issues
    headings_labels = 0 if pd.isna(headings_labels) else headings_labels

    # Check website accessibility
    website_status = check_website_status(url)

    # Define importance based on error frequency and severity
    important_features = {
        'Errors': total_errors,
        'Contrast Issues': contrast_issues,
        'Alerts': alerts,
        'Missing Alt Text': missing_alt_text,
        'Keyboard Accessibility Issues': keyboard_issues,
        'Headings and Labels': headings_labels
    }

    # Ensure all values are numbers before sorting
    sorted_features = sorted(important_features.items(), key=lambda x: float(x[1]), reverse=True)
    top_issues = [f"{k}: {v}" for k, v in sorted_features if v > 0]

    # Summary message
    message = f"Institution: {institution}\nDomain: {domain}\nWebsite: {url}\n{website_status}\n"
    message += f"Total Errors: {total_errors}, Contrast Issues: {contrast_issues}, Alerts: {alerts}\n"

    if top_issues:
        message += "Warning for other accessibility concerns:\n" + "\n".join(top_issues)
    else:
        message += "No other major accessibility problems detected."

    if observation_labels:
        message += f"\nAdditional Observations: {observation_labels}"

    return message


# Main function
def main():
    file_path = "C:/Users/USER/PycharmProjects/alfie/uc-3-data-explanation/csv-json/uc3-processed-data.xlsx"
    df_a, df_b, df_c = load_data(file_path)

    while True:
        country = input("Enter a country name (or type 'exit' to quit): ")
        if country.lower() == 'exit':
            break
        print(evaluate_accessibility(country, df_a))


if __name__ == "__main__":
    main()