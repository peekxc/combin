async function main() {
  // 1. Initialize Pyodide
  let pyodide = await loadPyodide();
  console.log("Pyodide Ready");

  // 2. Find all run buttons in your document
  const runButtons = document.querySelectorAll(".run-py-btn");

  runButtons.forEach(button => {
    button.addEventListener("click", async () => {
      // Find the associated code block and output div
      const codeId = button.getAttribute("data-code-id");
			console.log("Looking for code id: ", codeId);
      const codeElement = document.getElementById(codeId);
			console.log("Found code element: ", codeElement);
      const outputElement = document.getElementById(codeId + "-output");
			console.log("Found output element: ", outputElement);
      outputElement.innerText = "Running...";

      try {
        // 3. Execute the Python code
        let result = await pyodide.runPythonAsync(codeElement.innerText);
        outputElement.innerText = result || "Execution successful (no output)";
      } catch (err) {
        outputElement.innerText = `Error: ${err.message}`;
      }
    });
  });
}

// Start initialization
main();