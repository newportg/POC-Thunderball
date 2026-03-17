#!/usr/bin/env python3
"""
Automated script to fetch latest Thunderball results from National Lottery API
and update the local CSV file with new draws.
"""

import csv
from datetime import datetime
from pathlib import Path

import requests
from lxml import etree

# Suppress SSL warnings for development
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_FILE = Path("data/thunderball-draw-history.csv")
XML_URL = "https://www.national-lottery.co.uk/results/thunderball/draw-history/xml"


def fetch_xml():
    """Fetch XML from National Lottery."""
    response = requests.get(XML_URL, timeout=10, verify=False)
    response.raise_for_status()
    return response.content


def parse_draws(xml_content):
    """Parse XML and extract draw results."""
    root = etree.fromstring(xml_content)
    draws = []

    # XPath to find all draw elements
    draw_elements = root.xpath("//draw")

    for draw_elem in draw_elements:
        draw_num = draw_elem.xpath("draw-number/text()")
        draw_date = draw_elem.xpath("draw-date/text()")
        machine = draw_elem.xpath("draw-machine/text()")

        balls = draw_elem.xpath("../balls")
        if not balls:
            continue

        ball_set = balls[0].xpath("set/text()")
        ball_nums = balls[0].xpath("ball/text()")
        thunderball = balls[0].xpath("bonus-ball[@type='thunderball']/text()")

        if not (draw_num and draw_date and ball_nums and thunderball):
            continue

        # Convert date format: YYYY-MM-DD -> DD-Mon-YYYY
        try:
            date_obj = datetime.strptime(draw_date[0], "%Y-%m-%d")
            formatted_date = date_obj.strftime("%d-%b-%Y")
        except (ValueError, IndexError):
            continue

        draws.append(
            {
                "DrawDate": formatted_date,
                "Ball 1": ball_nums[0] if len(ball_nums) > 0 else "",
                "Ball 2": ball_nums[1] if len(ball_nums) > 1 else "",
                "Ball 3": ball_nums[2] if len(ball_nums) > 2 else "",
                "Ball 4": ball_nums[3] if len(ball_nums) > 3 else "",
                "Ball 5": ball_nums[4] if len(ball_nums) > 4 else "",
                "Thunderball": thunderball[0],
                "Ball Set": ball_set[0] if ball_set else "",
                "Machine": machine[0] if machine else "",
                "DrawNumber": draw_num[0],
            }
        )

    return draws


def read_existing_draws():
    """Read existing draw numbers from CSV."""
    if not DATA_FILE.exists():
        return set()

    existing = set()
    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("DrawNumber"):
                existing.add(row["DrawNumber"])

    return existing


def update_csv(new_draws):
    """Update CSV with new draws (prepend to keep latest first)."""
    if not new_draws:
        print("No new draws to add.")
        return

    existing_draws = read_existing_draws()
    draws_to_add = [d for d in new_draws if d["DrawNumber"] not in existing_draws]

    if not draws_to_add:
        print("All draws already in CSV.")
        return

    # Read existing data
    existing_data = []
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

    # Write new data (new draws first, then existing)
    fieldnames = [
        "DrawDate",
        "Ball 1",
        "Ball 2",
        "Ball 3",
        "Ball 4",
        "Ball 5",
        "Thunderball",
        "Ball Set",
        "Machine",
        "DrawNumber",
    ]

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for draw in draws_to_add:
            writer.writerow(draw)
        for row in existing_data:
            writer.writerow(row)

    print(f"✓ Added {len(draws_to_add)} new draw(s) to {DATA_FILE}")
    for draw in draws_to_add:
        print(
            f"  Draw {draw['DrawNumber']}: {draw['DrawDate']} | "
            f"{draw['Ball 1']}-{draw['Ball 2']}-{draw['Ball 3']}-{draw['Ball 4']}-{draw['Ball 5']} TB:{draw['Thunderball']}"
        )


def main():
    print(f"Fetching Thunderball results from {XML_URL}...")
    try:
        xml_content = fetch_xml()
        draws = parse_draws(xml_content)
        print(f"Parsed {len(draws)} total draws from XML.")
        update_csv(draws)
    except requests.RequestException as e:
        print(f"✗ Error fetching XML: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ Error processing draws: {e}")
        exit(1)


if __name__ == "__main__":
    main()
