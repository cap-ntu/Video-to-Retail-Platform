function toggle_process_region() {
    var process_region = document.getElementById("process-region");
    var checkbox = document.getElementsByName("if_process");
    process_region.style.display = (process_region.style.display === "none")? "" : "none";
}

function toggle_model_selection() {
    var selection_region = document.getElementById("model-selection");
    var checkbox = document.getElementsByName("if_default_model_selection");
    selection_region.style.display = (selection_region.style.display === "none")? "" : "none";
}