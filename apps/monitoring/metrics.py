from prometheus_client import Gauge, start_http_server
import time

STATUS = Gauge('mlops_job_status', 'Heartbeat for each MLOps job')


def run():
    start_http_server(9191)
    while True:
        STATUS.set(1)
        time.sleep(5)


if __name__ == '__main__':
    run()
