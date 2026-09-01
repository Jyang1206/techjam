import sys
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs = HfFileSystem()
for idx in sys.argv[1:]:
    path = f"datasets/OwensLab/CommunityForensics-Small/data/HFCF_small_{idx}.parquet"
    try:
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            md = pf.metadata
            li = md.schema.to_arrow_schema().get_field_index("label")
            vals = set()
            for rg in range(md.num_row_groups):
                st = md.row_group(rg).column(li).statistics
                if st is not None and st.has_min_max:
                    vals.add(st.min); vals.add(st.max)
            print(idx, md.num_rows, sorted(vals), flush=True)
    except Exception as e:
        print(idx, "ERR", repr(e)[:100], flush=True)
