import os
import shutil
import contextlib

@contextlib.contextmanager
def memory_report():
    # Setup: Clean directories and set environment variables
    xla_dumps_dir_path = '/tmp/xla_dumps'
    jax_dumps_dir_path = '/tmp/jax_dumps'
    shutil.rmtree(xla_dumps_dir_path, ignore_errors=True)
    shutil.rmtree(jax_dumps_dir_path, ignore_errors=True)
    
    original_xla_flags = os.environ.get("XLA_FLAGS")
    original_jax_dump_to = os.environ.get("JAX_DUMP_TO")

    os.environ["XLA_FLAGS"] = f"--xla_dump_to={xla_dumps_dir_path}"
    os.environ["JAX_DUMP_TO"] = jax_dumps_dir_path
    
    # Ensure directories exist for dumping
    os.makedirs(xla_dumps_dir_path, exist_ok=True)
    os.makedirs(jax_dumps_dir_path, exist_ok=True)

    try:
        yield
    finally:
        # Teardown: Generate and print the report
        if os.path.exists(xla_dumps_dir_path):
            for filename in os.listdir(xla_dumps_dir_path):
                if "memory-usage-report" in filename:
                    file_path = os.path.join(xla_dumps_dir_path, filename)
                    with open(file_path, 'r') as file:
                        file_lines = file.read()
                        print("\n\n")
                        print("============================ Memory Report ============================")
                        print(file_lines)
                        break
            else:
                print("\n\n")
                print("============================ Memory Report ============================")
                print("No memory usage report found.")

        # Restore original environment variables
        if original_xla_flags is None:
            if "XLA_FLAGS" in os.environ: # Check if it was set by us
                del os.environ["XLA_FLAGS"]
        else:
            os.environ["XLA_FLAGS"] = original_xla_flags
        
        if original_jax_dump_to is None:
            if "JAX_DUMP_TO" in os.environ: # Check if it was set by us
                del os.environ["JAX_DUMP_TO"]
        else:
            os.environ["JAX_DUMP_TO"] = original_jax_dump_to
        
        shutil.rmtree(xla_dumps_dir_path, ignore_errors=True)
        shutil.rmtree(jax_dumps_dir_path, ignore_errors=True)
