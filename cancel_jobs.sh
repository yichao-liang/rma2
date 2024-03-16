# Get all jobs
JOB_IDS=$(squeue -h -o "%A" -u ycliang)

# Loop over all jobs
for JOB_ID in $JOB_IDS
do
    # Get the name of the job
    JOB_NAME=$(squeue -h -o "%j" -j $JOB_ID)

    # Check if the job name is "bash"
    if [ "$JOB_NAME" != "bash" ]
    then
        # If it's not "bash", cancel the job
        scancel $JOB_ID
    fi
done
