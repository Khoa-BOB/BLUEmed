#!/usr/bin/env python3
"""
Build both Mayo Clinic and WebMD databases with rate limiting.
Processes in batches with delays to avoid hitting API rate limits.
"""
from config.settings import settings
from app.rag.vector_builder import MedicalVectorBuilder
from pathlib import Path
import time

print("="*80)
print("Building Medical Knowledge Vector Databases")
print("="*80)
print(f"Embedding model: {settings.EMBEDDING_MODEL}")
print(f"Persist directory: {settings.PERSIST_DIR}")
print(f"API Key: {'✅ Set' if settings.GOOGLE_API_KEY_EMBED else '❌ Not set'}")
print()

if not settings.GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not set in .env file")
    exit(1)

print("This will process both databases in batches with delays to avoid rate limits.")
print("Estimated time: 30-60 minutes (depending on API rate limits)")
print()

# Initialize builder
builder = MedicalVectorBuilder(embedding_model=settings.EMBEDDING_MODEL)

# Define paths
mayo_data = Path("BlueMed_data/MayoClinic")
webmd_data = Path("BlueMed_data/WebMD")
persist_base = Path(settings.PERSIST_DIR)

mayo_persist = persist_base / "mayo_clinic"
webmd_persist = persist_base / "webmd"

mayo_success = False
webmd_success = False

# Build Mayo Clinic database
print("\n" + "="*80)
print("STEP 1/2: Building Mayo Clinic Database")
print("="*80)

start_time = time.time()

mayo_db = builder.build_vectorstore(
    data_dir=mayo_data,
    persist_dir=mayo_persist,
    collection_name="mayo_clinic",
    source_name="mayo_clinic"
)

mayo_success = mayo_db is not None
mayo_time = time.time() - start_time

if mayo_success:
    print(f"\n✅ Mayo Clinic database built successfully in {mayo_time/60:.1f} minutes!")
else:
    print(f"\n❌ Mayo Clinic database failed to build")

# Wait between databases to avoid rate limits
if mayo_success:
    print("\n⏳ Waiting 30 seconds before building WebMD database...")
    time.sleep(30)

# Build WebMD database
print("\n" + "="*80)
print("STEP 2/2: Building WebMD Database")
print("="*80)

start_time = time.time()

webmd_db = builder.build_vectorstore(
    data_dir=webmd_data,
    persist_dir=webmd_persist,
    collection_name="webmd",
    source_name="webmd"
)

webmd_success = webmd_db is not None
webmd_time = time.time() - start_time

if webmd_success:
    print(f"\n✅ WebMD database built successfully in {webmd_time/60:.1f} minutes!")
else:
    print(f"\n❌ WebMD database failed to build")

# Summary
print("\n" + "="*80)
print("BUILD SUMMARY")
print("="*80)
print(f"Mayo Clinic: {'✅ Success' if mayo_success else '❌ Failed'}")
print(f"WebMD:       {'✅ Success' if webmd_success else '❌ Failed'}")
print()

if mayo_success and webmd_success:
    print("🎉 All databases built successfully!")
    print()
    print("Next steps:")
    print("  1. Set FORCE_REBUILD=False in .env to avoid rebuilding")
    print("  2. Run: python test_rag_autobuild.py")
    print("  3. Or run: python test_metadata_filtering.py")
elif mayo_success or webmd_success:
    print("⚠️  Partial success. You can use the databases that were built.")
    print("   Run the script again to retry failed databases.")
else:
    print("❌ Build failed. Check the error messages above.")
    print("   Common issues:")
    print("   - API rate limits (wait and retry)")
    print("   - Invalid API key")
    print("   - Network connection issues")
