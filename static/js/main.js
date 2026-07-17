// MediAssist — symptom picker interactions

(function () {
  const filter = document.getElementById("symptom-filter");
  const grid = document.getElementById("symptom-grid");
  const counter = document.getElementById("selected-count");
  if (!grid) return;

  const chips = Array.from(grid.querySelectorAll(".chip"));

  // Live filter
  if (filter) {
    filter.addEventListener("input", () => {
      const q = filter.value.trim().toLowerCase();
      chips.forEach((chip) => {
        const label = chip.textContent.trim().toLowerCase();
        chip.classList.toggle("hidden", q && !label.includes(q));
      });
    });
  }

  // Selection counter
  function updateCount() {
    const n = grid.querySelectorAll("input:checked").length;
    counter.textContent = n + (n === 1 ? " symptom selected" : " symptoms selected");
  }
  grid.addEventListener("change", updateCount);
  updateCount();

  // Prevent submitting with no symptom selected
  const form = document.querySelector(".intake-form");
  if (form) {
    form.addEventListener("submit", (e) => {
      if (grid.querySelectorAll("input:checked").length === 0) {
        e.preventDefault();
        counter.textContent = "Please select at least one symptom.";
        counter.style.color = "#b23a2e";
        grid.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  }
})();
