// Cache DOM element handles
const repoInput = document.getElementById("repo-input");
const syncBtn = document.getElementById("sync-btn");
const syncStatusBox = document.getElementById("sync-status-box");
const statusText = document.getElementById("status-text");
const syncDetails = document.getElementById("sync-details");
const currentRepoTitle = document.getElementById("current-repo-title");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const landingScreen = document.getElementById("landing-screen");
const landingRepoInput = document.getElementById("landing-repo-input");
const landingSyncBtn = document.getElementById("landing-sync-btn");
const collapseSidebarBtn = document.getElementById("collapse-sidebar-btn");
const sidebar = document.querySelector(".sidebar");
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const sidebarOverlay = document.getElementById("sidebar-overlay");

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("mobile-open");
        if (sidebarOverlay) sidebarOverlay.classList.toggle("active");
    });
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener("click", () => {
        sidebar.classList.remove("mobile-open");
        sidebarOverlay.classList.remove("active");
    });
}

// Initial welcome content cache to allow chat resetting
const initialWelcomeHtml = chatMessages.innerHTML;

// ==========================================================================
// 1. Textarea Auto-Resize & Input Listeners
// ==========================================================================
chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = (chatInput.scrollHeight - 4) + "px";
    
    // Enable/disable send button
    sendBtn.disabled = !chatInput.value.trim();
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim()) {
            submitUserQuestion();
        }
    }
});

sendBtn.addEventListener("click", () => {
    if (chatInput.value.trim()) {
        submitUserQuestion();
    }
});

// ==========================================================================
// 2. Repository Sync Pipeline (POST /sync)
// ==========================================================================
async function performSync(repo) {
    if (!repo) {
        updateSyncStatus("error", "Invalid Name", "Please enter a valid owner/repo name.");
        return false;
    }

    // Toggle loading states
    repoInput.disabled = true;
    syncBtn.disabled = true;
    updateSyncStatus("syncing", "Syncing Repository...", `Connecting to GitHub API to index '${repo}'...`);

    try {
        const response = await fetch(`/sync?repo=${encodeURIComponent(repo)}`, {
            method: "POST"
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        
        if (data.status === "success") {
            const timeStr = new Date().toLocaleTimeString();
            updateSyncStatus(
                "success", 
                "Sync Successful", 
                `<strong>Summary:</strong><br>` +
                `• Documents Skipped (Unchanged): ${data.docs_skipped || 0}<br>` +
                `• Documents Added (New): ${data.docs_added || 0}<br>` +
                `• Documents Updated (Mutated): ${data.docs_updated || 0}<br>` +
                `• Documents Deleted (Removed): ${data.docs_deleted || 0}<br>` +
                `• Total Examined: ${data.total_documents || 0}<br>` +
                `<small style="color: var(--text-muted)">Synced at ${timeStr}</small>`
            );
            
            // Update active header repo title
            currentRepoTitle.textContent = repo;
            return true;
        } else {
            updateSyncStatus("error", "Sync Execution Failed", data.message || "An unknown error occurred during sync.");
            return false;
        }

    } catch (err) {
        updateSyncStatus("error", "Sync Error", err.message);
        return false;
    } finally {
        repoInput.disabled = false;
        syncBtn.disabled = false;
    }
}

syncBtn.addEventListener("click", async () => {
    const repo = repoInput.value.trim();
    await performSync(repo);
});

function updateSyncStatus(state, headline, detailHtml) {
    syncStatusBox.className = `status-box ${state}`;
    statusText.textContent = headline;
    syncDetails.innerHTML = detailHtml;
}

// ==========================================================================
// 3. RAG Answer Pipeline (POST /ask)
// ==========================================================================
async function submitUserQuestion() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Clear input box
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    // 1. Render User Message
    appendMessage("user", parseMarkdown(query));

    // 2. Render Loading Indicator
    const loaderId = appendLoaderMessage();

    // Scroll to bottom
    scrollToBottom();

    let activeRepo = currentRepoTitle.textContent;
    if (activeRepo === "Select a Repository" || !activeRepo.trim()) {
        activeRepo = null;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 40000);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query, repo: activeRepo }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.message || errData.answer || `Server returned status code: ${response.status}`);
        }

        const data = await response.json();
        
        // 3. Render Assistant Response
        removeLoaderMessage(loaderId);
        appendAssistantResponse(data);

    } catch (err) {
        clearTimeout(timeoutId);
        removeLoaderMessage(loaderId);
        const errorMsg = err.name === "AbortError" ? "Request timed out while contacting the RAG backend. Please try again." : err.message;
        appendMessage("assistant", `<p style="color: var(--accent-red)"><i class="fa-solid fa-triangle-exclamation"></i> <strong>Error retrieving answer:</strong> ${errorMsg}</p>`);
    } finally {
        removeLoaderMessage(loaderId);
        scrollToBottom();
    }
}

// ==========================================================================
// 4. Dom Manipulation Helpers
// ==========================================================================
function appendMessage(sender, htmlContent) {
    const msgElement = document.createElement("div");
    msgElement.className = `message ${sender}`;
    
    const icon = sender === "user" ? "fa-user" : "fa-robot";
    const avatarColorClass = sender === "user" ? "avatar" : "avatar";
    
    msgElement.innerHTML = `
        <div class="${avatarColorClass}"><i class="fa-solid ${icon}"></i></div>
        <div class="message-content">${htmlContent}</div>
    `;
    
    chatMessages.appendChild(msgElement);
}

function appendLoaderMessage() {
    const loaderId = "loader-" + Date.now();
    const msgElement = document.createElement("div");
    msgElement.className = "message assistant";
    msgElement.id = loaderId;
    
    msgElement.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="message-content">
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(msgElement);
    return loaderId;
}

function removeLoaderMessage(id) {
    const loader = document.getElementById(id);
    if (loader) {
        loader.remove();
    }
}

function appendAssistantResponse(data) {
    const msgElement = document.createElement("div");
    msgElement.className = "message assistant";
    
    let responseHtml = parseMarkdown(data.answer);
    
    // Add clickable citation links if present
    if (data.source_chunks && data.source_chunks.length > 0) {
        const citationsList = [];
        
        // Filter sources based on which ones the model explicitly cited
        const citedIds = new Set(data.cited_chunk_ids || []);
        
        data.source_chunks.forEach(src => {
            // Note: If model did not cite any, default to showing top ones
            const isCited = citedIds.size === 0 || citedIds.has(src.chunk_id);
            
            if (isCited && src.url && src.url !== "N/A") {
                let iconClass = "fa-book";
                let label = "README Docs";
                
                if (src.doc_type === "pr") {
                    iconClass = "fa-code-merge";
                    const prNum = src.doc_id.split("-")[1] || "";
                    label = prNum ? `PR #${prNum}` : "Pull Request";
                } else if (src.doc_type === "issue") {
                    iconClass = "fa-circle-dot";
                    const issueNum = src.doc_id.split("-")[1] || "";
                    label = issueNum ? `Issue #${issueNum}` : "GitHub Issue";
                }
                
                citationsList.push(`
                    <a href="${src.url}" target="_blank" class="citation-card" title="${src.title || ''}">
                        <i class="fa-solid ${iconClass}"></i> ${label}
                    </a>
                `);
            }
        });
        
        if (citationsList.length > 0) {
            responseHtml += `
                <div class="citations-box">
                    <div class="citations-title"><i class="fa-solid fa-bookmark"></i> Citations</div>
                    <div class="citations-list">
                        ${citationsList.join("")}
                    </div>
                </div>
            `;
        }
    }
    
    msgElement.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="message-content">${responseHtml}</div>
    `;
    
    chatMessages.appendChild(msgElement);
}

// Markdown Parser Helper
function parseMarkdown(text) {
    if (!text) return "";
    
    // 1. Escape HTML
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // 2. Code blocks
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
        return `
            <div class="code-block-container">
                <button class="copy-code-btn"><i class="fa-regular fa-copy"></i> Copy</button>
                <pre><code>${code.trim()}</code></pre>
            </div>
        `;
    });

    // 3. Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 4. Bold text
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 5. Convert newlines to paragraphs
    html = html.split('\n\n').map(p => {
        if (p.includes('class="code-block-container"')) return p; // Skip wrapping preformatted blocks
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
}

function scrollToBottom() {
    const wrapper = chatMessages.parentElement;
    wrapper.scrollTop = wrapper.scrollHeight;
}

// ==========================================================================
// 5. Welcoming Prompt Suggestion Chips
// ==========================================================================
document.addEventListener("click", (e) => {
    if (e.target && e.target.classList.contains("chip")) {
        const question = e.target.getAttribute("data-question");
        if (question) {
            chatInput.value = question;
            chatInput.style.height = "auto";
            chatInput.style.height = (chatInput.scrollHeight - 4) + "px";
            sendBtn.disabled = false;
            submitUserQuestion();
        }
    }
});

// ==========================================================================
// 6. Clear/Reset Conversation History
// ==========================================================================
clearChatBtn.addEventListener("click", () => {
    const messages = chatMessages.querySelectorAll(".message");
    messages.forEach(msg => {
        msg.classList.add("fade-out");
    });
    setTimeout(() => {
        chatMessages.innerHTML = initialWelcomeHtml;
    }, 300);
});

// ==========================================================================
// 7. Light / Dark Theme Toggler
// ==========================================================================
function initTheme() {
    const savedTheme = localStorage.getItem("repomind-theme") || "dark";
    setTheme(savedTheme);
}

function setTheme(theme) {
    if (theme === "light") {
        document.documentElement.setAttribute("data-theme", "light");
        themeToggleBtn.innerHTML = '<i class="fa-solid fa-sun"></i>';
        localStorage.setItem("repomind-theme", "light");
    } else {
        document.documentElement.removeAttribute("data-theme");
        themeToggleBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
        localStorage.setItem("repomind-theme", "dark");
    }
}

themeToggleBtn.addEventListener("click", () => {
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    setTheme(isLight ? "dark" : "light");
});

// Landing Sync & Chat event handler
landingSyncBtn.addEventListener("click", async () => {
    const repo = landingRepoInput.value.trim();
    if (repo) {
        repoInput.value = repo;
        
        // Show loading state
        document.getElementById("landing-main-content").style.display = "none";
        document.getElementById("landing-loading-content").style.display = "flex";
        
        await performSync(repo);
        
        // When finished, show the chatbot interface
        landingScreen.classList.add("hidden");
    } else {
        landingScreen.classList.add("hidden");
    }
});

// Popular landing demo chips click handler
document.addEventListener("click", (e) => {
    const chip = e.target.closest(".landing-chip");
    if (chip) {
        const repo = chip.getAttribute("data-repo");
        if (repo) {
            landingRepoInput.value = repo;
            landingSyncBtn.click();
        }
    }
});

// Toggle Collapse/Expand Sidebar Panel
collapseSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    const isCollapsed = sidebar.classList.contains("collapsed");
    
    if (isCollapsed) {
        collapseSidebarBtn.innerHTML = '<i class="fa-solid fa-angles-right"></i>';
    } else {
        collapseSidebarBtn.innerHTML = '<i class="fa-solid fa-angles-left"></i>';
    }
});

// Copy to Clipboard Delegated Handler for Code Blocks
document.addEventListener("click", async (e) => {
    const copyBtn = e.target.closest(".copy-code-btn");
    if (copyBtn) {
        const container = copyBtn.closest(".code-block-container");
        const codeElement = container.querySelector("code");
        const textToCopy = codeElement.innerText;
        
        try {
            await navigator.clipboard.writeText(textToCopy);
            copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            copyBtn.classList.add("copied");
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                copyBtn.classList.remove("copied");
            }, 2000);
        } catch (err) {
            console.error("Failed to copy code: ", err);
        }
    }
});

// Run theme setup on load
initTheme();
