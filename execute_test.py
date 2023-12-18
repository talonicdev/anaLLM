import subprocess


def run_test():
    command = './venv/bin/python3.11 ./quality_test.py '

    p = subprocess.Popen(
            [command],
            shell=True,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True)
    out, err = p.communicate()

    with open('./results.txt', 'wb') as f:
        f.write(out)


run_test()
