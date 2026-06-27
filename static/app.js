document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Auth & Profile
    const loginScreen = document.getElementById("login-screen");
    const loginForm = document.getElementById("login-form");
    const loginUsername = document.getElementById("login-username");
    const loginPassword = document.getElementById("login-password");
    const loginError = document.getElementById("login-error");
    const appDashboard = document.getElementById("app-dashboard");
    const dietitianNameDisplay = document.getElementById("dietitian-name-display");
    const logoutBtn = document.getElementById("logout-btn");
    
    // DOM Elements - Patient Details Form
    const form = document.getElementById("diet-form");
    const patientNameInput = document.getElementById("patient-name");
    const weightInput = document.getElementById("weight");
    const heightInput = document.getElementById("height");
    const bmiValDisplay = document.getElementById("bmi-display-val");
    const bmiBadge = document.getElementById("bmi-badge-display");
    const fillDemoBtn = document.getElementById("fill-demo");
    const submitBtn = document.getElementById("submit-btn");
    const printBtn = document.getElementById("print-btn");
    
    const placeholderBox = document.getElementById("placeholder-box");
    const resultsBox = document.getElementById("results-box");
    const dietPlanContent = document.getElementById("diet-plan-content");
    
    const loadingSpinner = document.getElementById("loading-spinner");
    const loadingStepText = document.getElementById("loading-step-text");
    const loadingProgressBar = document.getElementById("loading-progress");

    // Configure marked options for secure and attractive HTML output
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    // 1. Authentication State Management
    function checkAuth() {
        const token = localStorage.getItem("ayurdiet_token");
        const dietitianName = localStorage.getItem("ayurdiet_dietitian");
        if (token) {
            loginScreen.style.display = "none";
            appDashboard.style.display = "block";
            dietitianNameDisplay.textContent = dietitianName || "Dr. Sarah (Dietitian)";
        } else {
            loginScreen.style.display = "flex";
            appDashboard.style.display = "none";
        }
    }
    
    // Execute auth check on load
    checkAuth();

    // 1b. Google Sign-In Integration
    // The client ID is injected server-side by Flask directly into the page
    function initGoogleSignIn() {
        const clientId = window.__GOOGLE_CLIENT_ID__;
        if (!clientId || clientId.trim() === "" || clientId === "__GOOGLE_CLIENT_ID__") return;

        // Show divider and Google Sign-in button
        document.getElementById("login-divider").style.display = "flex";
        document.getElementById("google-signin-btn").style.display = "flex";

        const checkAndInit = () => {
            if (typeof google !== "undefined" && google.accounts && google.accounts.id) {
                google.accounts.id.initialize({
                    client_id: clientId,
                    callback: handleGoogleCredentialResponse
                });
                google.accounts.id.renderButton(
                    document.getElementById("google-signin-btn"),
                    { theme: "outline", size: "large", width: 370 }
                );
            } else {
                setTimeout(checkAndInit, 100);
            }
        };
        checkAndInit();
    }

    async function handleGoogleCredentialResponse(googleResponse) {
        loginError.style.display = "none";
        try {
            const response = await fetch("/api/login-google", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ credential: googleResponse.credential })
            });
            const data = await response.json();
            
            if (response.ok && data.success) {
                localStorage.setItem("ayurdiet_token", data.token);
                localStorage.setItem("ayurdiet_dietitian", data.dietitian_name);
                checkAuth();
            } else {
                loginError.style.display = "block";
                loginError.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${data.error || "Google Authentication failed"}`;
            }
        } catch (error) {
            loginError.style.display = "block";
            loginError.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Unable to verify Google session.`;
            console.error(error);
        }
    }

    // Load Google authentication options
    initGoogleSignIn();

    // Login Form Handler
    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        loginError.style.display = "none";
        
        const username = loginUsername.value.trim();
        const password = loginPassword.value.trim();
        
        try {
            const response = await fetch("/api/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            
            if (response.ok && data.success) {
                localStorage.setItem("ayurdiet_token", data.token);
                localStorage.setItem("ayurdiet_dietitian", data.dietitian_name);
                checkAuth();
                
                // Clear login fields
                loginUsername.value = "";
                loginPassword.value = "";
            } else {
                loginError.style.display = "block";
                loginError.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${data.error || "Invalid username or password"}`;
            }
        } catch (error) {
            loginError.style.display = "block";
            loginError.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Unable to connect to server.`;
            console.error(error);
        }
    });

    // Logout Handler
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("ayurdiet_token");
        localStorage.removeItem("ayurdiet_dietitian");
        checkAuth();
        
        // Reset patient details and view state
        form.reset();
        patientNameInput.value = "";
        updateBMI();
        placeholderBox.style.display = "flex";
        resultsBox.style.display = "none";
        printBtn.setAttribute("disabled", "true");
    });

    // 2. Calculate BMI Dynamically
    function updateBMI() {
        const weight = parseFloat(weightInput.value);
        const height = parseFloat(heightInput.value);
        
        if (weight > 0 && height > 0) {
            const heightM = height / 100;
            const bmi = weight / (heightM * heightM);
            const bmiFormatted = bmi.toFixed(1);
            
            bmiValDisplay.textContent = bmiFormatted;
            
            // Set badge category and class
            bmiBadge.className = "bmi-badge"; // Reset classes
            if (bmi < 18.5) {
                bmiBadge.textContent = "Underweight";
                bmiBadge.classList.add("bmi-underweight");
            } else if (bmi < 25) {
                bmiBadge.textContent = "Normal";
                bmiBadge.classList.add("bmi-normal");
            } else if (bmi < 30) {
                bmiBadge.textContent = "Overweight";
                bmiBadge.classList.add("bmi-overweight");
            } else {
                bmiBadge.textContent = "Obese";
                bmiBadge.classList.add("bmi-obese");
            }
        } else {
            bmiValDisplay.textContent = "--";
            bmiBadge.textContent = "No Data";
            bmiBadge.className = "bmi-badge";
        }
    }

    weightInput.addEventListener("input", updateBMI);
    heightInput.addEventListener("input", updateBMI);

    // 3. Pre-fill Demo Data
    fillDemoBtn.addEventListener("click", () => {
        patientNameInput.value = "Rahul Sharma";
        document.getElementById("prakriti").value = "Pitta";
        document.getElementById("age").value = "25";
        document.getElementById("gender").value = "Female";
        weightInput.value = "58";
        heightInput.value = "162";
        document.getElementById("activity").value = "moderate";
        document.getElementById("sleep").value = "irregular";
        document.getElementById("stress").value = "high";
        document.getElementById("region").value = "South India";
        document.getElementById("season").value = "Summer";
        document.getElementById("preferences").value = "Vegetarian, dairy-free";
        document.getElementById("deficiency").value = "iron";
        document.getElementById("health").value = "Anaemia";
        
        updateBMI();
        
        // Dynamic notification toast effect in button
        const originalText = fillDemoBtn.innerHTML;
        fillDemoBtn.innerHTML = `<i class="fa-solid fa-check"></i> Filled!`;
        fillDemoBtn.style.color = "#0d9488";
        setTimeout(() => {
            fillDemoBtn.innerHTML = originalText;
            fillDemoBtn.style.color = "";
        }, 1500);
    });

    // 4. Form Submission & Generation
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Prepare request payload
        const payload = {
            patient_name: patientNameInput.value.trim(),
            prakriti: document.getElementById("prakriti").value,
            age: parseInt(document.getElementById("age").value),
            gender: document.getElementById("gender").value,
            weight: parseFloat(weightInput.value),
            height: parseFloat(heightInput.value),
            activity: document.getElementById("activity").value,
            sleep: document.getElementById("sleep").value,
            stress: document.getElementById("stress").value,
            region: document.getElementById("region").value,
            season: document.getElementById("season").value,
            preferences: document.getElementById("preferences").value || "None",
            nutrient_deficiency: document.getElementById("deficiency").value,
            health: document.getElementById("health").value || "None"
        };

        // Show loading screen
        loadingSpinner.style.display = "flex";
        loadingProgressBar.style.width = "0%";
        
        // Progress steps emulation
        const steps = [
            { text: "Reading biometrics and mapping Prakriti constitution...", progress: "15%" },
            { text: "Querying FAISS Vector Database for Ayurvedic rule embeddings...", progress: "40%" },
            { text: "Searching Neo4j Aura Cloud Graph for seasonal and regional foods...", progress: "70%" },
            { text: "Processing hybrid context and generating diet plan with Google Gemini...", progress: "90%" }
        ];

        let currentStep = 0;
        loadingStepText.textContent = steps[0].text;
        loadingProgressBar.style.width = steps[0].progress;

        const intervalId = setInterval(() => {
            currentStep++;
            if (currentStep < steps.length) {
                loadingStepText.textContent = steps[currentStep].text;
                loadingProgressBar.style.width = steps[currentStep].progress;
            }
        }, 2200);

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            clearInterval(intervalId);

            if (response.ok && data.success) {
                loadingProgressBar.style.width = "100%";
                loadingStepText.textContent = "Plan generated successfully!";
                
                setTimeout(() => {
                    // Hide loader
                    loadingSpinner.style.display = "none";
                    
                    // Show results box, hide placeholder
                    placeholderBox.style.display = "none";
                    resultsBox.style.display = "block";
                    
                    // Render Markdown
                    dietPlanContent.innerHTML = marked.parse(data.diet_plan);
                    
                    // Enable print button
                    printBtn.removeAttribute("disabled");
                    
                    // Smooth scroll to results on smaller viewports
                    resultsBox.scrollIntoView({ behavior: "smooth" });
                }, 500);

            } else {
                throw new Error(data.error || data.details || "Unknown API server error");
            }

        } catch (error) {
            clearInterval(intervalId);
            loadingSpinner.style.display = "none";
            
            // Format a nice error alert inside the output panel
            placeholderBox.style.display = "none";
            resultsBox.style.display = "block";
            printBtn.setAttribute("disabled", "true");
            
            dietPlanContent.innerHTML = `
                <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 1.5rem; color: #ef4444;">
                    <h3 style="color: #dc2626; margin-top: 0; display: flex; align-items: center; gap: 0.5rem; font-family: 'Outfit', sans-serif;">
                        <i class="fa-solid fa-triangle-exclamation"></i> Generation Failed
                    </h3>
                    <p style="margin-bottom: 0.75rem; font-size: 0.95rem;">An error occurred while compiling the patient's Ayurvedic diet plan. Please verify server status and try again.</p>
                    <code style="display: block; background: #fee2e2; padding: 0.75rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #b91c1c;">
                        Error: ${error.message}
                    </code>
                </div>
            `;
        }
    });

    // 5. Print / PDF action
    printBtn.addEventListener("click", () => {
        if (!printBtn.hasAttribute("disabled")) {
            window.print();
        }
    });
});
