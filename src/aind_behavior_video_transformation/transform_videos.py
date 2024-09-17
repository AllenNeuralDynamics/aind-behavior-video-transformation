"""Module to handle transforming behavior videos"""

import logging
import sys
from time import time

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)


class JobSettings(BasicJobSettings):
    """BehaviorJob settings. Inherits both fields input_source and
    output_directory from BasicJobSettings."""


class BehaviorVideoJob(GenericEtl[JobSettings]):
    """Main class to handle behavior video transformations"""

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job
        Returns
        -------
        JobResponse
          Information about the job that can be used for metadata downstream.

        """
        job_start_time = time()
        # Code to process data stored in self.job_settings.input_source
        # and write to self.job_settings.output_directory
        job_end_time = time()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time-job_start_time}",
            data=None,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = JobSettings.model_validate_json(cli_args.job_settings)
    elif cli_args.config_file is not None:
        job_settings = JobSettings.from_config_file(cli_args.config_file)
    else:
        # Construct settings from env vars
        job_settings = JobSettings()
    job = BehaviorVideoJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())
