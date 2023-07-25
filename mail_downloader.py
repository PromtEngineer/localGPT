import os
import imaplib
import email
import pdfkit
from bs4 import BeautifulSoup
import subprocess
import argparse
from datetime import datetime


# IMAP configuration
IMAP_SERVER = 'your_imap_server'
USERNAME = 'your_email@example.com'
# PASSWORD = 'your_email_password'
MAILBOX = 'INBOX'


# PDF output directory
OUTPUT_DIR = 'SOURCE_DOCUMENTS'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def get_emails():
    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(USERNAME, PASSWORD)
    mail.select(MAILBOX)

    # Search for all emails
    result, data = mail.uid('search', None, 'ALL')
    if result != 'OK':
        print("Error searching emails.")
        return []

    email_ids = data[0].split()
    return email_ids

def remove_images_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for img_tag in soup.find_all('img'):
        img_tag.extract()
    return str(soup)

def extract_text_html(msg):
    content_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                content_parts.append(part.get_payload(decode=True))
    else:
        content_type = msg.get_content_type()
        if content_type == "text/html":
            content_parts.append(msg.get_payload(decode=True))

    return b"\n".join(content_parts)

def get_available_filename(base_filename, folder):
    name, ext = os.path.splitext(base_filename)
    index = 1
    new_filename = base_filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name} ({index}){ext}"
        index += 1
    return new_filename


def download_emails_as_pdf():
    email_ids = get_emails()
    if not email_ids:
        print("No emails found.")
        return

    for email_id in email_ids:
        result, data = mail.uid('fetch', email_id, '(RFC822)')
        if result == 'OK':
            raw_email = data[0][1]

            # Parse the email using the email library
            msg = email.message_from_bytes(raw_email)
            
            # Get the date of the email
            if args.password is None:
                date_obj = email.utils.parsedate_to_datetime(msg['Date'])
                date_str = date_obj.strftime('%Y-%m-%d')
            else:
                date_str=''
            

            # Get the subject of the email to use as the PDF filename
            subject = msg.get("Subject", "No Subject")
            pdf_filename = f"{date_str}_{subject}.pdf"
            pdf_filename = get_available_filename(pdf_filename, OUTPUT_DIR)


            # Extract text/html parts from the email content
            email_html = extract_text_html(msg)
            email_html = remove_images_from_html(email_html)

            # Save the email content as a PDF file
            #pdfkit.from_string(email_html, pdf_filename)
            pdfkit.from_string(email_html, os.path.join(OUTPUT_DIR, pdf_filename))


            print(f"Saved email '{subject}' as PDF: {pdf_filename}")
    
    # Run script2.py from script1.py
    
    result = subprocess.run(['python3', 'ingest.py'], capture_output=True, text=True)
    print(result.stdout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download emails and save as PDF.')
    parser.add_argument('--server', default=IMAP_SERVER, help='IMAP server address')
    parser.add_argument('--username', default=USERNAME, help='Email username')
    parser.add_argument('--mailbox', default=MAILBOX, help='Mailbox to fetch emails from')
    parser.add_argument('--password', default=PASSWORD, help='Email password')

    args = parser.parse_args()

    IMAP_SERVER = args.server
    USERNAME = args.username
    MAILBOX = args.mailbox

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    if args.password is None:
        args.password = input("Password: ")
    else:
        args.password = PASSWORD

    mail.login(USERNAME, args.password)
    mail.select(MAILBOX)
    download_emails_as_pdf()
    mail.logout()


