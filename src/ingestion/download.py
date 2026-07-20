"""
scripts/data_downloader.py

Downloads SEC 10-K filings for a list of companies from SEC EDGAR
and saves them into data/raw/sec_filings/.

Run:
    python -m scripts.data_downloader
"""
import sys
from pathlib import Path

# Make sure "src" is importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import DocumentLoader

# ---- Configure which companies you want here ----
# CIK numbers are unique per company on SEC EDGAR
COMPANIES = {
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    "Amazon": "0001018724",
    "Alphabet": "0001652044",
    "Tesla": "0001318605",
}

DEST_DIR = "data/raw/sec_filings"
FORMS = ("10-K",)          # add "10-Q", "8-K" here if you want those too
FILINGS_PER_COMPANY = 1    # how many recent filings to grab per company


def download_all():
    loader = DocumentLoader()
    downloaded = []

    for company_name, cik in COMPANIES.items():
        print(f"\nFetching filings list for {company_name} (CIK {cik})...")
        try:
            filings = loader.fetch_edgar_filings(cik=cik, forms=FORMS)
        except Exception as e:
            print(f"  FAILED to fetch filing list: {e}")
            continue

        if not filings:
            print(f"  No {FORMS} filings found for {company_name}")
            continue

        for filing in filings[:FILINGS_PER_COMPANY]:
            print(f"  Downloading {filing['form']} filed {filing['filing_date']}: "
                  f"{filing['primary_document']}")
            try:
                path = loader.download_filing(
                    cik=cik,
                    accession_number=filing["accession_number"],
                    primary_document=filing["primary_document"],
                    dest_dir=DEST_DIR,
                )
                downloaded.append({
                    "company": company_name,
                    "path": str(path),
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                })
                print(f"    Saved -> {path}")
            except Exception as e:
                print(f"    FAILED to download: {e}")

    print(f"\nDone. {len(downloaded)} filings downloaded to {DEST_DIR}/")
    return downloaded


if __name__ == "__main__":
    download_all()