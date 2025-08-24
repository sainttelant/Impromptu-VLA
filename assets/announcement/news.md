# Recent Updates to Code and Evaluation

We have recently made updates to our codebase and evaluation methodology to ensure the accuracy and reproducibility of our results. The previous version of our code contained an issue that led to incorrect results, and we sincerely apologize for this oversight.

The following changes have been implemented:

Multi-View Input: The closed-loop evaluation now uses three-view images as input, rather than a single view. Additionally, we have restricted the data to only use information from the past 3 seconds.

Improved Evaluation Scripts: We have updated the evaluation scripts so that files are now automatically generated based on time. This change prevents data from different experiments from getting mixed up when running multiple tests simultaneously, a problem that caused errors in the previous version.

Increased Evaluation Runs: To ensure the stability and reliability of our results, we have increased the number of closed-loop evaluation runs from 10 to 50.

Due to these changes, the results in our README have been updated. We found that the previous Base+nuScenes result was incorrect and have now replaced it with the accurate result from the 50-run evaluation. This new result is lower than before, confirming the previous error.

In contrast, the Base+Impromptu+nuScenes result remains largely unchanged. This reinforces that the use of our data leads to a more significant improvement in closed-loop performance.

Please use these updated results as the correct reference. We are committed to maintaining a high standard of research and thank you for your understanding.