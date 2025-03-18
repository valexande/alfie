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
        if response.status_code != 404:
            return True
        else:
            print(url, response.status_code)
            return False
    except requests.RequestException:
        return True


def get_continent_statistics(df_a):
    """Generate accessibility statistics for each continent, including website accessibility check."""
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
                          "Cayman Islands",
                          "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala",
                          "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Puerto Rico", "Saint Lucia",
                          "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States of America"],
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

        # Check website accessibility
        failed_countries = []
        for _, row in continent_data.iterrows():
            if not check_website_status(row['URL']):
                failed_countries.append(row['Country'])

        statistics[continent] = {
            "Total Websites": total_websites,
            "Total Errors": total_errors,
            "Total Contrast Issues": total_contrast_issues,
            "Total Alerts": total_alerts,
            "Average Errors per Website": round(avg_errors, 2),
            "Average Contrast Issues per Website": round(avg_contrast_issues, 2),
            "Average Alerts per Website": round(avg_alerts, 2)
        }

        if failed_countries:
            inaccessible_websites[continent] = failed_countries

    return statistics, inaccessible_websites


def main():
    file_path = "C:/Users/USER/PycharmProjects/alfie/uc-3-data-explanation/csv-json/uc3-processed-data.xlsx"
    df_a, df_b, df_c = load_data(file_path)
    statistics, inaccessible_websites = get_continent_statistics(df_a)

    for continent, stats in statistics.items():
        print(f"\nStatistics for {continent}:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\nWebsites that failed accessibility check:")
    for continent, countries in inaccessible_websites.items():
        print(f"{continent}: {', '.join(countries)}")


if __name__ == "__main__":
    main()
