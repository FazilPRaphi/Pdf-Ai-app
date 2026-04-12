(() => {
    const MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024;
    const UPLOAD_TIMEOUT_MS = 8 * 60 * 1000;

    const state = {
        selectedFile: null,
        uploadedDocuments: [],
        activeDocumentId: "",
        isUploading: false,
        isAsking: false,
    };

    const elements = {
        dropZone: document.getElementById("dropZone"),
        dropTitle: document.getElementById("dropTitle"),
        dropHint: document.getElementById("dropHint"),
        fileInput: document.getElementById("fileInput"),
        browseButton: document.getElementById("browseButton"),
        uploadButton: document.getElementById("uploadButton"),
        uploadStatus: document.getElementById("uploadStatus"),
        uploadProgressWrap: document.getElementById("uploadProgressWrap"),
        uploadProgressFill: document.getElementById("uploadProgressFill"),
        uploadProgressLabel: document.getElementById("uploadProgressLabel"),
        uploadProgressText: document.getElementById("uploadProgressText"),
        documentSelect: document.getElementById("documentSelect"),
        activeDocumentBadge: document.getElementById("activeDocumentBadge"),
        chatMessages: document.getElementById("chatMessages"),
        chatForm: document.getElementById("chatForm"),
        questionInput: document.getElementById("questionInput"),
        sendButton: document.getElementById("sendButton"),
        quickActions: document.getElementById("quickActions"),
        chatHelper: document.getElementById("chatHelper"),
        toastContainer: document.getElementById("toastContainer"),
    };

    init();

    function init() {
        bindUploadEvents();
        bindChatEvents();
        addAssistantMessage("Upload a PDF to start asking grounded questions.", []);
        updateDocumentSelect();
        updateChatControls();
    }

    function bindUploadEvents() {
        elements.browseButton.addEventListener("click", () => elements.fileInput.click());
        elements.fileInput.addEventListener("change", () => {
            const file = elements.fileInput.files && elements.fileInput.files[0];
            if (file) {
                setSelectedFile(file);
            }
        });

        elements.uploadButton.addEventListener("click", () => {
            void uploadSelectedFile();
        });

        ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
            elements.dropZone.addEventListener(eventName, preventDefaults, false);
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            elements.dropZone.addEventListener(eventName, () => {
                elements.dropZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            elements.dropZone.addEventListener(eventName, () => {
                elements.dropZone.classList.remove("dragover");
            });
        });

        elements.dropZone.addEventListener("drop", (event) => {
            const droppedFile = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
            if (droppedFile) {
                setSelectedFile(droppedFile);
            }
        });

        elements.documentSelect.addEventListener("change", (event) => {
            state.activeDocumentId = event.target.value;
            updateChatControls();
            updateActiveDocumentBadge();
        });
    }

    function bindChatEvents() {
        elements.chatForm.addEventListener("submit", (event) => {
            event.preventDefault();
            void askQuestion();
        });
        elements.quickActions.addEventListener("click", (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }
            const action = target.dataset.action;
            if (!action) {
                return;
            }
            handleQuickAction(action);
        });

        elements.questionInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                elements.chatForm.requestSubmit();
            }
        });

        elements.questionInput.addEventListener("input", autoResizeQuestionInput);
    }

    function setSelectedFile(file) {
        const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
        if (!isPdf) {
            showToast("Only PDF files are allowed.", "error");
            return;
        }
        if (file.size > MAX_UPLOAD_SIZE_BYTES) {
            const maxSizeMb = Math.round(MAX_UPLOAD_SIZE_BYTES / (1024 * 1024));
            showToast(`File is too large. Maximum supported size is ${maxSizeMb} MB.`, "error");
            return;
        }

        state.selectedFile = file;
        elements.dropTitle.textContent = file.name;
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        elements.dropHint.textContent = `Ready to upload (${sizeMb} MB)`;
        elements.uploadButton.disabled = state.isUploading;
    }

    async function uploadSelectedFile() {
        if (!state.selectedFile || state.isUploading) {
            return;
        }

        setUploadingState(true);
        setUploadProgress(0);
        setUploadProgressLabel("Uploading...");
        toggleProgress(true);
        renderProcessingStatus(`Uploading ${state.selectedFile.name}...`);
        showToast("Large PDF processing may take 1-2 mins.", "success");

        const formData = new FormData();
        formData.append("file", state.selectedFile);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload-pdf", true);
        xhr.timeout = UPLOAD_TIMEOUT_MS;

        xhr.upload.addEventListener("progress", (event) => {
            if (!event.lengthComputable) {
                return;
            }
            const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
            setUploadProgress(percent);
            setUploadProgressLabel("Uploading...");
        });
        xhr.upload.addEventListener("load", () => {
            setUploadProgress(100);
            setUploadProgressLabel("Processing PDF...");
            renderProcessingStatus("Processing PDF... this can take up to 1-2 mins for large files.");
        });

        xhr.onerror = () => {
            setUploadingState(false);
            toggleProgress(false);
            const message = "Upload failed due to a network error.";
            renderUploadError(message);
            showToast(message, "error");
        };

        xhr.ontimeout = () => {
            setUploadingState(false);
            toggleProgress(false);
            const message = "Upload timed out while processing PDF. Please retry.";
            renderUploadError(message);
            showToast(message, "error");
        };

        xhr.onreadystatechange = () => {
            if (xhr.readyState !== XMLHttpRequest.DONE) {
                return;
            }

            setUploadingState(false);

            let payload = {};
            try {
                payload = JSON.parse(xhr.responseText || "{}");
            } catch (_error) {
                payload = {};
            }

            if (xhr.status >= 200 && xhr.status < 300 && payload.success) {
                setUploadProgress(100);
                handleUploadSuccess(payload);
                window.setTimeout(() => toggleProgress(false), 400);
                return;
            }

            toggleProgress(false);
            const errorMessage = formatUploadError(payload, xhr.status);
            showToast(errorMessage, "error");
            renderUploadError(errorMessage);
        };

        xhr.send(formData);
    }

    function handleUploadSuccess(payload) {
        const documentInfo = {
            document_id: payload.document_id,
            filename: payload.filename,
            pages: payload.pages,
            upload_date: payload.upload_date,
            total_chunks_stored: payload.total_chunks_stored,
        };

        state.uploadedDocuments = [
            documentInfo,
            ...state.uploadedDocuments.filter((doc) => doc.document_id !== documentInfo.document_id),
        ];
        state.activeDocumentId = documentInfo.document_id;

        state.selectedFile = null;
        elements.fileInput.value = "";
        elements.dropTitle.textContent = "Drag and drop a PDF";
        elements.dropHint.textContent = "or click to browse local files";

        updateDocumentSelect();
        renderUploadStatus(documentInfo);
        updateChatControls();
        updateActiveDocumentBadge();

        showToast(`Uploaded ${documentInfo.filename} successfully.`, "success");
    }

    async function askQuestion() {
        if (state.isAsking || !state.activeDocumentId) {
            return;
        }

        const question = elements.questionInput.value.trim();
        if (!question) {
            return;
        }

        addUserMessage(question);
        elements.questionInput.value = "";
        autoResizeQuestionInput();

        const loadingEntry = addLoadingMessage();
        setAskingState(true);

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    question,
                    document_id: state.activeDocumentId,
                }),
            });

            const payload = await response.json();
            loadingEntry.remove();

            if (!response.ok || !payload.success) {
                const errorMessage = payload.error || payload.message || "Could not process your question.";
                addAssistantMessage(errorMessage, []);
                showToast(errorMessage, "error");
                return;
            }

            addAssistantMessage(payload.answer || "No answer was returned.", payload.sources || []);
        } catch (_error) {
            loadingEntry.remove();
            const errorMessage = "Something went wrong while asking your question.";
            addAssistantMessage(errorMessage, []);
            showToast(errorMessage, "error");
        } finally {
            setAskingState(false);
        }
    }

    function handleQuickAction(action) {
        if (!state.activeDocumentId || state.isAsking || state.isUploading) {
            return;
        }

        const actionMap = {
            summarize: "Summarize this file in 5-7 lines.",
            "key-points": "List the key points from this file in bullet points.",
            skills: "Extract skills, requirements, and qualifications from this file.",
        };
        const question = actionMap[action];
        if (!question) {
            return;
        }
        elements.questionInput.value = question;
        autoResizeQuestionInput();
        elements.chatForm.requestSubmit();
    }

    function addUserMessage(text) {
        const entry = createChatEntry("user");
        const bubble = document.createElement("div");
        bubble.className = "bubble user";
        bubble.textContent = text;
        entry.appendChild(bubble);
        elements.chatMessages.appendChild(entry);
        scrollChatToBottom();
    }

    function addAssistantMessage(answerText, sources) {
        const entry = createChatEntry("assistant");
        const bubble = document.createElement("div");
        bubble.className = "bubble assistant";

        const answer = document.createElement("div");
        answer.className = "answer-text";
        answer.innerHTML = formatText(answerText);
        bubble.appendChild(answer);

        if (Array.isArray(sources) && sources.length > 0) {
            const sourceToggle = document.createElement("details");
            sourceToggle.className = "sources-toggle";

            const summary = document.createElement("summary");
            summary.textContent = `View Sources (${sources.length})`;
            sourceToggle.appendChild(summary);

            const sourceList = document.createElement("ul");
            sourceList.className = "sources-list";

            sources.forEach((source) => {
                const item = document.createElement("li");
                const filename = source.filename || "Unknown file";
                const heading = source.heading ? ` - ${source.heading}` : "";
                const preview = source.text ? `: ${source.text}` : "";
                item.textContent = `${filename}${heading}${preview}`;
                sourceList.appendChild(item);
            });

            sourceToggle.appendChild(sourceList);
            bubble.appendChild(sourceToggle);
        }

        entry.appendChild(bubble);
        elements.chatMessages.appendChild(entry);
        scrollChatToBottom();
    }

    function addLoadingMessage() {
        const entry = createChatEntry("assistant");
        const bubble = document.createElement("div");
        bubble.className = "bubble assistant";
        bubble.innerHTML = '<span class="typing"><span class="spinner"></span>Thinking...</span>';
        entry.appendChild(bubble);
        elements.chatMessages.appendChild(entry);
        scrollChatToBottom();
        return entry;
    }

    function createChatEntry(role) {
        const entry = document.createElement("div");
        entry.className = `chat-entry ${role}`;
        return entry;
    }

    function updateDocumentSelect() {
        elements.documentSelect.innerHTML = "";

        if (state.uploadedDocuments.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No uploaded document";
            elements.documentSelect.appendChild(option);
            elements.documentSelect.disabled = true;
            updateActiveDocumentBadge();
            return;
        }

        state.uploadedDocuments.forEach((doc) => {
            const option = document.createElement("option");
            option.value = doc.document_id;
            option.textContent = doc.filename;
            option.selected = doc.document_id === state.activeDocumentId;
            elements.documentSelect.appendChild(option);
        });

        elements.documentSelect.disabled = false;
        updateActiveDocumentBadge();
    }

    function updateActiveDocumentBadge() {
        if (!state.activeDocumentId) {
            elements.activeDocumentBadge.textContent = "No document selected";
            return;
        }

        const activeDoc = state.uploadedDocuments.find(
            (doc) => doc.document_id === state.activeDocumentId,
        );

        elements.activeDocumentBadge.textContent = activeDoc
            ? `Active: ${activeDoc.filename}`
            : "No document selected";
    }

    function renderUploadStatus(doc) {
        elements.uploadStatus.classList.remove("empty");
        elements.uploadStatus.innerHTML = `
            <div class="status-row"><span>Filename</span><strong>${escapeHtml(doc.filename)}</strong></div>
            <div class="status-row"><span>Document ID</span><strong>${escapeHtml(doc.document_id)}</strong></div>
            <div class="status-row"><span>Pages</span><strong>${escapeHtml(String(doc.pages || "n/a"))}</strong></div>
            <div class="status-row"><span>Uploaded</span><strong>${escapeHtml(doc.upload_date || "now")}</strong></div>
            <div class="status-row"><span>Chunks Stored</span><strong>${escapeHtml(String(doc.total_chunks_stored))}</strong></div>
        `;
    }

    function renderProcessingStatus(message) {
        elements.uploadStatus.classList.remove("empty");
        elements.uploadStatus.innerHTML = `
            <span class="status-inline"><span class="spinner"></span>${escapeHtml(message)}</span>
        `;
    }

    function renderUploadError(message) {
        elements.uploadStatus.classList.remove("empty");
        elements.uploadStatus.innerHTML = `<span>${escapeHtml(message)}</span>`;
    }

    function setUploadingState(isUploading) {
        state.isUploading = isUploading;
        elements.uploadButton.disabled = isUploading || !state.selectedFile;
        elements.browseButton.disabled = isUploading;
        elements.fileInput.disabled = isUploading;
        elements.dropZone.classList.toggle("disabled", isUploading);
        elements.documentSelect.disabled = isUploading || state.uploadedDocuments.length === 0;
    }

    function setAskingState(isAsking) {
        state.isAsking = isAsking;
        updateChatControls();
    }

    function updateChatControls() {
        const hasDocument = Boolean(state.activeDocumentId);
        elements.questionInput.disabled = !hasDocument || state.isAsking || state.isUploading;
        elements.sendButton.disabled = !hasDocument || state.isAsking || state.isUploading;
        const quickButtons = elements.quickActions.querySelectorAll(".btn-quick");
        quickButtons.forEach((button) => {
            button.disabled = !hasDocument || state.isAsking || state.isUploading;
        });

        if (!hasDocument) {
            elements.questionInput.placeholder = "Upload a PDF first to ask questions";
            elements.chatHelper.textContent = "Upload and select a document to enable chat.";
        } else if (state.isUploading) {
            elements.questionInput.placeholder = "Please wait for PDF processing to complete";
            elements.chatHelper.textContent = "Processing PDF...";
        } else {
            elements.questionInput.placeholder = "Ask a grounded question about the selected PDF";
            elements.chatHelper.textContent = "";
        }
    }

    function toggleProgress(visible) {
        if (visible) {
            elements.uploadProgressWrap.classList.add("visible");
            return;
        }
        elements.uploadProgressWrap.classList.remove("visible");
        setUploadProgress(0);
    }

    function setUploadProgress(percent) {
        const safePercent = Math.max(0, Math.min(100, percent));
        elements.uploadProgressFill.style.width = `${safePercent}%`;
        elements.uploadProgressText.textContent = `${safePercent}%`;
    }

    function setUploadProgressLabel(text) {
        elements.uploadProgressLabel.textContent = text;
    }

    function autoResizeQuestionInput() {
        elements.questionInput.style.height = "auto";
        elements.questionInput.style.height = `${elements.questionInput.scrollHeight}px`;
    }

    function showToast(message, type) {
        const toast = document.createElement("div");
        toast.className = `toast ${type === "error" ? "error" : "success"}`;
        toast.textContent = message;

        elements.toastContainer.appendChild(toast);

        window.setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(-8px)";
            window.setTimeout(() => toast.remove(), 180);
        }, 3000);
    }

    function scrollChatToBottom() {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    function formatText(text) {
        return escapeHtml(text)
            .replace(/\n\n/g, "<br><br>")
            .replace(/\n/g, "<br>");
    }

    function escapeHtml(text) {
        return String(text)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function preventDefaults(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    function formatUploadError(payload, statusCode) {
        if (payload && (payload.error || payload.message)) {
            return payload.error || payload.message;
        }
        if (statusCode === 413) {
            return "File is too large. Please upload a smaller PDF.";
        }
        if (statusCode >= 500) {
            return "Failed to process PDF. Please retry.";
        }
        return `Upload failed (HTTP ${statusCode}).`;
    }
})();
