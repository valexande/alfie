from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# First Endpoint: Evaluate Accessibility by Country
@app.route('/evaluate-country-accessibility', methods=['POST'])
def evaluate_country():
    excel_file = request.files['excel_file']
    country = request.form['country']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(excel_file.filename))
    excel_file.save(file_path)

    df_a, df_b, df_c = load_data(file_path)
    result = evaluate_accessibility(country, df_a)
    return jsonify({'result': result})

# Second Endpoint: Continent-Wise Accessibility Stats
@app.route('/continent-accessibility-summary', methods=['POST'])
def continent_summary():
    excel_file = request.files['excel_file']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(excel_file.filename))
    excel_file.save(file_path)

    df_a, df_b, df_c = load_data(file_path)
    statistics, inaccessible = get_continent_statistics(df_a)
    return jsonify({'statistics': statistics, 'inaccessible_websites': inaccessible})

# Shared utility functions
def load_data(file_path):
    df_a = pd.read_excel(file_path, sheet_name='a-part-of-data')
    df_b = pd.read_excel(file_path, sheet_name='b-part-of-data', header=None)
    df_c = pd.read_excel(file_path, sheet_name='c-part-of-data', header=None)
    return df_a, df_b, df_c

def check_website_status(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code < 300
    except requests.RequestException:
        return False

def evaluate_accessibility(country, df_a):
    country_data = df_a[df_a['Country'].str.lower() == country.lower()]
    if country_data.empty:
        return f"No data available for {country}."

    row = country_data.iloc[0]
    url = row['URL']
    institution = row['Institution']
    domain = row['Domain']

    total_errors = pd.to_numeric(row['Errors'], errors='coerce') or 0
    contrast_issues = pd.to_numeric(row['Contrast Errors'], errors='coerce') or 0
    alerts = pd.to_numeric(row['Alerts'], errors='coerce') or 0
    missing_alt_text = pd.to_numeric(row['Missing alternative text'], errors='coerce') or 0
    keyboard_issues = pd.to_numeric(row['Keyboard (Level A)'], errors='coerce') or 0
    headings_labels = pd.to_numeric(row['Headings and Labels (Level AA)'], errors='coerce') or 0
    observation_labels = row['Observation'] if isinstance(row['Observation'], str) else "No specific observations."

    website_status = "accessible" if check_website_status(url) else "not accessible"

    important_features = {
        'Errors': total_errors,
        'Contrast Issues': contrast_issues,
        'Alerts': alerts,
        'Missing Alt Text': missing_alt_text,
        'Keyboard Accessibility Issues': keyboard_issues,
        'Headings and Labels': headings_labels
    }

    sorted_features = sorted(important_features.items(), key=lambda x: float(x[1]), reverse=True)
    top_issues = [f"{k}: {v}" for k, v in sorted_features if v > 0]

    message = f"Institution: {institution}\nDomain: {domain}\nWebsite: {url}\nWebsite is {website_status}.\n"
    message += f"Total Errors: {total_errors}, Contrast Issues: {contrast_issues}, Alerts: {alerts}\n"

    if top_issues:
        message += "Warning for other accessibility concerns:\n" + "\n".join(top_issues)
    else:
        message += "No other major accessibility problems detected."

    message += f"\nAdditional Observations: {observation_labels}"
    return message

def get_continent_statistics(df_a):
    countries_by_continent = {
        "Africa": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon",
                   "Central African Republic", "Chad", "Comoros", "Congo", "Democratic Republic of Congo", "Djibouti",
                   "Egypt", "Equatorial Guinea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
                   "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
                   "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
                   "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia",
                   "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia",
                   "Zimbabwe"],
        "Asia": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei Darussalam",
                 "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan",
                 "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia",
                 "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
                 "Russia", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan",
                 "Tajikistan", "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", "United Arab Emirates",
                 "Uzbekistan", "Vietnam", "Yemen"],
        "Europe": ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
                   "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany",
                   "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
                   "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
                   "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
                   "Sweden", "Switzerland", "Ukraine", "United Kingdom"],
        "North America": ["Anguilla", "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada",
                          "Cayman Islands", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador",
                          "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama",
                          "Puerto Rico", "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago",
                          "United States of America"],
        "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay",
                          "Peru", "Suriname", "Uruguay", "Venezuela"],
        "Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand",
                    "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"]
    }
    statistics = {}
    inaccessible_websites = {}

    for continent, countries in countries_by_continent.items():
        continent_data = df_a[df_a['Country'].isin(countries)]
        if continent_data.empty:
            continue

        total_errors = continent_data['Errors'].sum()
        total_contrast_issues = continent_data['Contrast Errors'].sum()
        total_alerts = continent_data['Alerts'].sum()
        avg_errors = continent_data['Errors'].mean()
        avg_contrast_issues = continent_data['Contrast Errors'].mean()
        avg_alerts = continent_data['Alerts'].mean()
        total_websites = len(continent_data)

        failed_countries = [row['Country'] for _, row in continent_data.iterrows() if not check_website_status(row['URL'])]

        statistics[continent] = {
            "Total Websites": int(total_websites),
            "Total Errors": int(total_errors),
            "Total Contrast Issues": int(total_contrast_issues),
            "Total Alerts": int(total_alerts),
            "Average Errors per Website": round(float(avg_errors), 2),
            "Average Contrast Issues per Website": round(float(avg_contrast_issues), 2),
            "Average Alerts per Website": round(float(avg_alerts), 2)
        }

        if failed_countries:
            inaccessible_websites[continent] = failed_countries

    return statistics, inaccessible_websites

if __name__ == '__main__':
    app.run(port=5050, debug=True)