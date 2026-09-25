% Load the data from the spreadsheet
[data, ~, ~] = xlsread('ProcessingtimeResults.xlsx');

% Extract relevant columns
NumThreads = data(:, 1); % Column 1: Number of Threads
ExecutionTime = data(:, 2); % Column 2: Execution Time

% Plot Processing Time against Number of Threads
figure;
plot(NumThreads, ExecutionTime, '-o', 'DisplayName', 'Execution Time', 'LineWidth', 2, 'MarkerSize', 8);

% Add labels, title, and legend
title('Processing Time vs Number of Threads');
xlabel('Number of Threads');
ylabel('Processing Time (s)');
legend('show', 'Location', 'best');
grid on;

% Adjust Y-axis for better visualization if needed
ylim([0, max(ExecutionTime) * 1.1]); % Set Y-axis to 10% above max time
