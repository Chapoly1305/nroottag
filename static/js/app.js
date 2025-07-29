// Global state
let serverStatus = false;
let saladToken = localStorage.getItem('saladToken') || '';
let saladConfig = JSON.parse(localStorage.getItem('saladConfig') || '{}');

// API base URL
const API_BASE = window.location.origin;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
async function initializeApp() {
    logToConsole('Initializing nRootTag Control Center...', 'info');
    
    // Load saved configuration
    loadConfiguration();
    
    // Check server status
    await checkServerStatus();
    
    // Load current tasks
    await loadTasks();
    
    // Initial container status fetch
    await fetchContainerStatus();
    
    // Set up periodic updates
    setInterval(checkServerStatus, 5000);
    setInterval(loadTasks, 2000);
    setInterval(refreshResults, 2000);
    setInterval(fetchContainerStatus, 10000); // Refresh container status every 10 seconds
    
    logToConsole('System ready', 'success');
}

// Configuration Management
function loadConfiguration() {
    updateConfigurationStatus();
}

function updateConfigurationStatus() {
    const configStatus = document.getElementById('configStatus');
    const isConfigured = saladToken && saladConfig.orgName && saladConfig.projectName && saladConfig.containerGroup;
    
    if (isConfigured) {
        configStatus.classList.add('configured');
        configStatus.innerHTML = '<span><i class="fas fa-check-circle"></i> Configuration Complete</span>';
    } else {
        configStatus.classList.remove('configured');
        configStatus.innerHTML = '<span><i class="fas fa-exclamation-triangle"></i> Configuration Required</span>';
    }
}

function openConfigModal() {
    // Load current values into modal
    document.getElementById('modalSaladToken').value = saladToken || '';
    document.getElementById('modalOrgName').value = saladConfig.orgName || '';
    document.getElementById('modalProjectName').value = saladConfig.projectName || '';
    document.getElementById('modalContainerGroup').value = saladConfig.containerGroup || '';
    
    document.getElementById('configModal').style.display = 'block';
}

function saveConfiguration() {
    const token = document.getElementById('modalSaladToken').value.trim();
    const orgName = document.getElementById('modalOrgName').value.trim();
    const projectName = document.getElementById('modalProjectName').value.trim();
    const containerGroup = document.getElementById('modalContainerGroup').value.trim();
    
    if (!token || !orgName || !projectName || !containerGroup) {
        showNotification('Please fill in all configuration fields', 'error');
        return;
    }
    
    // Save to localStorage
    saladToken = token;
    saladConfig = { orgName, projectName, containerGroup };
    
    localStorage.setItem('saladToken', saladToken);
    localStorage.setItem('saladConfig', JSON.stringify(saladConfig));
    
    // Update status display
    updateConfigurationStatus();
    
    // Close modal
    closeModal('configModal');
    
    logToConsole('Configuration saved successfully', 'success');
    showNotification('Configuration saved successfully', 'success');
}

function clearConfiguration() {
    if (!confirm('Are you sure you want to clear all configuration data?')) {
        return;
    }
    
    // Clear localStorage
    localStorage.removeItem('saladToken');
    localStorage.removeItem('saladConfig');
    
    // Reset global variables
    saladToken = '';
    saladConfig = {};
    
    // Clear modal fields
    document.getElementById('modalSaladToken').value = '';
    document.getElementById('modalOrgName').value = '';
    document.getElementById('modalProjectName').value = '';
    document.getElementById('modalContainerGroup').value = '';
    
    // Update status display
    updateConfigurationStatus();
    
    logToConsole('Configuration cleared', 'info');
    showNotification('Configuration cleared', 'info');
}

// Server Status Management
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        serverStatus = data.continue;
        updateStatusIndicator(serverStatus);
        
    } catch (error) {
        updateStatusIndicator(false, true);
        logToConsole('Failed to check server status: ' + error.message, 'error');
    }
}


// Unified Search Management
async function handleSearch() {
    const searchInput = document.getElementById('searchInput');
    const input = searchInput.value.trim();
    
    if (!input) {
        showNotification('Please enter a hex string', 'error');
        return;
    }
    
    if (!/^[0-9a-fA-F]+$/.test(input)) {
        showNotification('Please enter a valid hex string (0-9, a-f, A-F only)', 'error');
        return;
    }
    
    if (input.length < 6 || input.length > 12) {
        showNotification('Please enter 6-12 characters', 'error');
        return;
    }
    
    if (input.length === 6) {
        // 6 characters - search for any address starting with this prefix
        // We'll search for prefix + "000000" as an example
        logToConsole(`Searching prefix ${input} range (using ${input}000000 as sample)`, 'info');
        await searchAddress(input + '000000');
    } else if (input.length === 12) {
        // 12 characters - search specific address
        await searchAddress(input);
    } else {
        showNotification('Please enter exactly 6 characters (prefix) or 12 characters (address)', 'error');
        return;
    }
    
    searchInput.value = '';
}

async function addTaskFromInput() {
    const searchInput = document.getElementById('searchInput');
    const input = searchInput.value.trim();
    
    if (!input) {
        showNotification('Please enter a hex string', 'error');
        return;
    }
    
    if (!/^[0-9a-fA-F]+$/.test(input)) {
        showNotification('Please enter a valid hex string (0-9, a-f, A-F only)', 'error');
        return;
    }
    
    if (input.length < 6 || input.length > 12) {
        showNotification('Please enter 6-12 characters', 'error');
        return;
    }
    
    let address = input; // Use input exactly as entered
    
    try {
        logToConsole(`Adding task for address: ${address}`, 'info');
        
        const response = await fetch(`${API_BASE}/public-key`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({address: address})
        });
        
        if (response.ok) {
            const data = await response.json();
            logToConsole(`Address ${address} found! Public key: ${data.public_key}`, 'success');
            showNotification('Address already exists! Check console and refresh results.', 'success');
            await refreshResults();
        } else if (response.status === 404) {
            const error = await response.json();
            logToConsole(`Address ${address} not found, added to search tasks: ${error.detail}`, 'info');
            showNotification('Address added to search tasks', 'success');
            await loadTasks();
        } else {
            const error = await response.json();
            logToConsole(`Failed to add task for ${address}: ${error.detail}`, 'error');
            showNotification('Failed to add task: ' + error.detail, 'error');
        }
        
    } catch (error) {
        logToConsole('Failed to add task: ' + error.message, 'error');
        showNotification('Failed to add task', 'error');
    }
    
    searchInput.value = '';
}

async function addSearchTask(prefix) {
    try {
        const response = await fetch(`${API_BASE}/add-search-task`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prefix: prefix})
        });
        
        if (response.ok) {
            await loadTasks();
            logToConsole(`Added search task: ${prefix}`, 'success');
            showNotification('Search task added successfully', 'success');
        }
    } catch (error) {
        logToConsole('Failed to add search task: ' + error.message, 'error');
    }
}

async function searchAddress(address) {
    try {
        logToConsole(`Searching for address: ${address}`, 'info');
        
        // Use /public-key endpoint which saves found keys to storage
        const response = await fetch(`${API_BASE}/public-key`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({address: address})
        });
        
        if (response.ok) {
            const data = await response.json();
            logToConsole(`Found public key for ${address}: ${data.public_key}`, 'success');
            showNotification('Address found! Check Search Results.', 'success');
            
            // Refresh results to show the newly saved key
            await refreshResults();
        } else if (response.status === 404) {
            const error = await response.json();
            logToConsole(`Address ${address} not found: ${error.detail}`, 'error');
            showNotification('Address not found in existing data', 'error');
        } else {
            const error = await response.json();
            logToConsole(`Search failed for ${address}: ${error.detail}`, 'error');
            showNotification('Search failed: ' + error.detail, 'error');
        }
    } catch (error) {
        logToConsole('Failed to search address: ' + error.message, 'error');
        showNotification('Search request failed', 'error');
    }
}

async function loadTasks() {
    try {
        const response = await fetch(`${API_BASE}/search-task`);
        const text = await response.text();
        
        const tasks = text.trim().split('\n').filter(t => t);
        const taskList = document.getElementById('taskList');
        
        if (tasks.length === 0) {
            taskList.innerHTML = '<div class="loading">No active tasks</div>';
        } else {
            taskList.innerHTML = tasks.slice(0, 10).map(task => 
                `<div class="task-item">${task}</div>`
            ).join('');
            
            if (tasks.length > 10) {
                taskList.innerHTML += `<div class="task-item">... and ${tasks.length - 10} more</div>`;
            }
        }
    } catch (error) {
        logToConsole('Failed to load tasks: ' + error.message, 'error');
    }
}

async function clearAllTasks() {
    if (!confirm('Are you sure you want to clear all search tasks?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/delete-search-task`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({key: '*'})
        });
        
        if (response.ok) {
            await loadTasks();
            logToConsole('All search tasks cleared', 'success');
            showNotification('All tasks cleared successfully', 'success');
        } else {
            const error = await response.json();
            logToConsole('Failed to clear tasks: ' + error.detail, 'error');
        }
    } catch (error) {
        logToConsole('Failed to clear tasks: ' + error.message, 'error');
    }
}

async function viewTasks() {
    try {
        const response = await fetch(`${API_BASE}/search-task`);
        const text = await response.text();
        
        const tasks = text.trim().split('\n').filter(t => t);
        const modalTaskList = document.getElementById('modalTaskList');
        
        modalTaskList.innerHTML = tasks.map(task => 
            `<div class="task-item">${task}</div>`
        ).join('');
        
        document.getElementById('taskModal').style.display = 'block';
    } catch (error) {
        logToConsole('Failed to load tasks: ' + error.message, 'error');
    }
}

// Salad Cloud Control
async function startContainers() {
    if (!validateSaladConfig()) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/saladcloud-start-containers`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                organization_name: saladConfig.orgName,
                project_name: saladConfig.projectName,
                container_group_name: saladConfig.containerGroup,
                salad_api_key: saladToken
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            logToConsole('Containers started successfully', 'success');
            showNotification('Containers started', 'success');
        } else {
            // Handle detailed error response
            if (data.success === false) {
                logToConsole(data.message, 'error');
                showNotification(data.message, 'error');
            } else {
                logToConsole('Failed to start containers: ' + JSON.stringify(data), 'error');
                showNotification('Failed to start containers', 'error');
            }
        }
    } catch (error) {
        logToConsole('Failed to start containers: ' + error.message, 'error');
    }
}

async function stopContainers() {
    if (!validateSaladConfig()) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/saladcloud-stop-containers`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                organization_name: saladConfig.orgName,
                project_name: saladConfig.projectName,
                container_group_name: saladConfig.containerGroup,
                salad_api_key: saladToken
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            logToConsole('Containers stopped successfully', 'success');
            showNotification('Containers stopped', 'success');
        } else {
            // Handle detailed error response
            if (data.success === false) {
                logToConsole(data.message, 'error');
                showNotification(data.message, 'error');
            } else {
                logToConsole('Failed to stop containers: ' + JSON.stringify(data), 'error');
                showNotification('Failed to stop containers', 'error');
            }
        }
    } catch (error) {
        logToConsole('Failed to stop containers: ' + error.message, 'error');
    }
}

// Results Management
async function refreshResults() {
    try {
        const response = await fetch(`${API_BASE}/review-storage`);
        const data = await response.json();
        
        const resultsContainer = document.getElementById('resultsContainer');
        const entries = Object.entries(data);
        const validEntries = entries.filter(([address, key]) => key && key !== '').reverse();
        
        if (validEntries.length === 0) {
            resultsContainer.innerHTML = '<div class="loading">No results found</div>';
        } else {
            // Get public keys for each found private key
            const resultsWithPublicKeys = await Promise.all(
                validEntries.map(async ([address, privateKey]) => {
                    try {
                        // Only try to get public key if address is 12 characters
                        if (address.length === 12) {
                            const pubResponse = await fetch(`${API_BASE}/public-key`, {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({address: address})
                            });
                            
                            if (pubResponse.ok) {
                                const pubData = await pubResponse.json();
                                if (pubData.public_key) {
                                    return {
                                        address,
                                        privateKey,
                                        publicKey: pubData.public_key
                                    };
                                }
                            }
                        }
                        // For non-12 char addresses or if public key fetch fails, just show the private key
                        return { address, privateKey, publicKey: 'N/A' };
                    } catch (error) {
                        return { address, privateKey, publicKey: 'Error' };
                    }
                })
            );
            
            resultsContainer.innerHTML = resultsWithPublicKeys.map(({address, privateKey, publicKey}) => `
                <div class="result-item">
                    <div class="result-row">
                        <span class="result-label">Addr:</span>
                        <span class="result-value result-address">${address}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">PrivKey:</span>
                        <span class="result-value result-private-key">${privateKey}</span>
                    </div>
                    <div class="result-row">
                        <span class="result-label">PubKey:</span>
                        <span class="result-value result-public-key">${publicKey}</span>
                    </div>
                </div>
            `).join('');
            
            // Only log when results change
            const currentCount = resultsContainer.dataset.resultCount || '0';
            if (validEntries.length.toString() !== currentCount) {
                resultsContainer.dataset.resultCount = validEntries.length.toString();
                if (validEntries.length > 0) {
                    logToConsole(`Found ${validEntries.length} results`, 'info');
                }
            }
        }
    } catch (error) {
        logToConsole('Failed to load results: ' + error.message, 'error');
        document.getElementById('resultsContainer').innerHTML = '<div class="loading">Error loading results</div>';
    }
}

async function exportResults() {
    try {
        const response = await fetch(`${API_BASE}/review-storage`);
        const data = await response.json();
        
        const csvContent = 'Address,Private Key\n' + 
            Object.entries(data)
                .filter(([address, key]) => key && key !== '')
                .map(([address, key]) => `${address},${key}`)
                .join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nroottag_results_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        logToConsole('Results exported successfully', 'success');
    } catch (error) {
        logToConsole('Failed to export results: ' + error.message, 'error');
    }
}

// Utility Functions
function validateSaladConfig() {
    if (!saladToken) {
        showNotification('Please configure your Salad Cloud token first', 'error');
        return false;
    }
    
    if (!saladConfig.orgName || !saladConfig.projectName || !saladConfig.containerGroup) {
        showNotification('Please complete all Salad Cloud configuration fields', 'error');
        return false;
    }
    
    return true;
}

function updateStatusIndicator(isActive, isError = false) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    
    dot.classList.remove('active', 'error');
    
    if (isError) {
        dot.classList.add('error');
        text.textContent = 'Connection Error';
    } else if (isActive) {
        dot.classList.add('active');
        text.textContent = 'Connected';
    } else {
        text.textContent = 'Server Stopped';
    }
}

function logToConsole(message, type = 'info') {
    const console = document.getElementById('console');
    const entry = document.createElement('div');
    entry.className = `console-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    console.appendChild(entry);
    console.scrollTop = console.scrollHeight;
    
    // Keep only last 50 entries
    while (console.children.length > 50) {
        console.removeChild(console.firstChild);
    }
}

function showNotification(message, type = 'info') {
    // Simple notification - you could enhance this with a toast library
    logToConsole(message, type);
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

// Click outside modal to close
window.onclick = function(event) {
    const taskModal = document.getElementById('taskModal');
    const configModal = document.getElementById('configModal');
    
    if (event.target === taskModal) {
        taskModal.style.display = 'none';
    }
    if (event.target === configModal) {
        configModal.style.display = 'none';
    }
}


// Enter key support for search input
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleSearch();
            }
        });
        
        // Update button text based on input length
        searchInput.addEventListener('input', function(e) {
            const input = e.target.value.trim();
            const buttonText = document.getElementById('searchButtonText');
            
            if (input.length === 6 && /^[0-9a-fA-F]{6}$/.test(input)) {
                buttonText.textContent = 'Search Prefix';
            } else if (input.length === 12 && /^[0-9a-fA-F]{12}$/.test(input)) {
                buttonText.textContent = 'Search Address';
            } else {
                buttonText.textContent = 'Search';
            }
        });
    }
});

// Container Status Functions
async function fetchContainerStatus() {
    if (!validateSaladConfig()) {
        const statusDiv = document.getElementById('containerStatus');
        statusDiv.innerHTML = '<div class="config-status-display"><i class="fas fa-exclamation-triangle"></i> Configuration Required</div>';
        return;
    }
    
    try {
        const params = new URLSearchParams({
            organization_name: saladConfig.orgName,
            project_name: saladConfig.projectName,
            container_group_name: saladConfig.containerGroup,
            salad_api_key: saladToken
        });
        
        const response = await fetch(`${API_BASE}/saladcloud-container-status?${params}`);
        const data = await response.json();
        
        const statusDiv = document.getElementById('containerStatus');
        
        if (data.success) {
            const group = data.container_group;
            statusDiv.innerHTML = `
                <div class="container-stats">
                    <div class="stat-item">
                        <div class="stat-label">Total</div>
                        <div class="stat-value">${group.total_instances}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Ready</div>
                        <div class="stat-value ready">${group.ready_instances}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Deploying</div>
                        <div class="stat-value deploying">${group.deploying_instances}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Failed</div>
                        <div class="stat-value failed">${group.failed_instances}</div>
                    </div>
                </div>
                <div class="container-details">
                    <div class="detail-row">
                        <span class="detail-label">Status:</span>
                        <span class="detail-value">${group.status || 'N/A'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Configured Replicas:</span>
                        <span class="detail-value">${group.replicas || 'N/A'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Current State:</span>
                        <span class="detail-value">${group.current_state?.status || 'N/A'}</span>
                    </div>
                </div>
            `;
        } else {
            statusDiv.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-circle"></i> ${data.message}</div>`;
        }
    } catch (error) {
        const statusDiv = document.getElementById('containerStatus');
        statusDiv.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-circle"></i> Failed to fetch container status</div>`;
        logToConsole('Failed to fetch container status: ' + error.message, 'error');
    }
}