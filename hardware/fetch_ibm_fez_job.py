# Retrieve the ibm_fez holonomy run (job d986q62f47jc73a7hm2g) from IBM Quantum and
# archive it as hardware/results/ibm_fez_holonomy_20260709.json (cited by note v3).
#
# The original JSON was written inside an ephemeral Colab session and not persisted;
# the raw results remain in the IBM Quantum account (IBM Quantum Platform on IBM
# Cloud) under the pinned job ID, subject to IBM's retention policy - retrieve while
# available. Run this in Colab (token via userdata, as in the multibackend notebook)
# or locally with QISKIT_IBM_TOKEN set in the environment.
#
# Pinned figures the archive must reproduce (note v3, Hardware paragraph):
#   loop phase -2.93 deg +/- 1.55 deg (ray +1; -i excluded ~56 sigma)
#   PM S = 4.6125 +/- 0.0173 (35.4 sigma over the tight deterministic bound 4)
import json, os, datetime

JOB_ID = "d986q62f47jc73a7hm2g"
_BASE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.getcwd())  # notebook-safe
OUT = os.path.join(_BASE, "results", "ibm_fez_holonomy_20260709.json")

def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN")
    crn = os.environ.get("QISKIT_IBM_CRN")
    if token is None:
        try:  # Colab
            from google.colab import userdata
            token = userdata.get("QISKIT_IBM_TOKEN")
            try:
                crn = userdata.get("QISKIT_IBM_CRN")
            except Exception:
                crn = None
        except Exception:
            pass
    if token is None:
        raise SystemExit("Set QISKIT_IBM_TOKEN (env or Colab userdata).")
    return QiskitRuntimeService(token=token, instance=crn) if crn else \
           QiskitRuntimeService(token=token)

def main():
    service = get_service()
    job = service.job(JOB_ID)
    res = job.result()
    payload = {
        "job_id": JOB_ID,
        "backend": getattr(job, "backend", lambda: None)() and job.backend().name,
        "retrieved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pinned_figures": {
            "loop_phase_deg": [-2.93, 1.55],
            "pm_S": [4.6125, 0.0173],
            "nc_bound": 4,
        },
        "results": [],
    }
    for i, pub in enumerate(res):
        entry = {"pub": i, "metadata": dict(getattr(pub, "metadata", {}) or {})}
        data = pub.data
        for field in data:
            bits = getattr(data, field)
            try:
                entry.setdefault("counts", {})[field] = bits.get_counts()
            except Exception:
                entry.setdefault("raw", {})[field] = str(bits)
        payload["results"].append(entry)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"archived {len(payload['results'])} pubs -> {OUT}")
    print("verify the pinned figures against the counts before committing.")

if __name__ == "__main__":
    main()
