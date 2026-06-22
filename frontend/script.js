/**
 * Bangla News Headline Generator - Frontend
 */

const API_BASE = window.location.origin;

// ===== Navigation =====
function navigateTo(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    // Remove active from nav links
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    // Show target page
    var target = document.getElementById('page-' + pageId);
    if (target) {
        target.classList.add('active');
    }

    // Set active nav link
    document.querySelectorAll('.nav-link').forEach(function(l) {
        if (l.textContent.trim().toLowerCase().replace(' ', '-') === pageId ||
            (pageId === 'home' && l.textContent.trim() === 'Home') ||
            (pageId === 'generator' && l.textContent.trim() === 'Generator') ||
            (pageId === 'model-info' && l.textContent.trim() === 'Model Info') ||
            (pageId === 'results' && l.textContent.trim() === 'Results') ||
            (pageId === 'about' && l.textContent.trim() === 'About')) {
            l.classList.add('active');
        }
    });

    // Close mobile menu
    document.getElementById('nav-links').classList.remove('show');

    // Load page data
    if (pageId === 'results') loadPredictions();
}

// ===== Generator =====
function loadSampleArticle() {
    document.getElementById('article-input').value = 'গাজীপুরের কালিয়াকৈর উপজেলার তেলিরচালা এলাকায় আজ বৃহস্পতিবার রাতের টিফিন খেয়ে একটি পোশাক কারখানার ৫০০ শ্রমিক অসুস্থ হয়ে পড়েছেন। এ ঘটনায় বিক্ষোভ করেছেন শ্রমিকেরা। স্থানীয় সূত্রে জানা গেছে, কালিয়াকৈর উপজেলার তেলিরচালা এলাকার একটি পোশাক কারখানায় আজ রাত সাড়ে আটটার দিকে কারখানার ক্যান্টিন থেকে সরবরাহ করা টিফিন খেয়ে প্রায় ৫০০ শ্রমিক অসুস্থ হয়ে পড়েন। এতে শ্রমিকদের মধ্যে বিক্ষোভের সৃষ্টি হয়। পরে অসুস্থ শ্রমিকদের বিভিন্ন হাসপাতাল ও ক্লিনিকে ভর্তি করা হয়েছে।';
    document.getElementById('actual-headline').value = 'কালিয়াকৈরে টিফিন খেয়ে ৫০০ শ্রমিক অসুস্থ, বিক্ষোভ';
}

async function generateHeadline() {
    var article = document.getElementById('article-input').value.trim();

    if (!article) {
        showError('অনুগ্রহ করে একটি বাংলা আর্টিকেল পেস্ট করুন।');
        return;
    }
    if (article.length < 10) {
        showError('আর্টিকেল খুব ছোট। কমপক্ষে ১০ অক্ষর দিন।');
        return;
    }

    document.getElementById('loading-spinner').style.display = 'block';
    document.getElementById('output-section').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('generate-btn').disabled = true;

    try {
        var payload = { article: article };
        var actual = document.getElementById('actual-headline').value.trim();
        if (actual) payload.actual_headline = actual;

        var res = await fetch(API_BASE + '/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            var err = await res.json();
            throw new Error(err.detail || 'Failed to generate headline.');
        }

        var data = await res.json();

        document.getElementById('generated-headline').textContent = data.headline;
        document.getElementById('output-device').textContent = data.device;
        document.getElementById('output-length').textContent = data.input_length;
        document.getElementById('output-time').textContent = data.timestamp;

        // Show actual headline comparison
        if (data.actual_headline) {
            document.getElementById('actual-headline-output').textContent = data.actual_headline;
            document.getElementById('actual-headline-display').style.display = 'block';

            // Calculate similarity score
            var score = calculateSimilarity(data.headline, data.actual_headline);
            showScore(score);
        } else {
            document.getElementById('actual-headline-display').style.display = 'none';
            document.getElementById('score-section').style.display = 'none';
        }

        document.getElementById('output-section').style.display = 'block';
    } catch (err) {
        showError(err.message);
    } finally {
        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('generate-btn').disabled = false;
    }
}

function calculateSimilarity(generated, actual) {
    // Simple word-overlap based score (similar to ROUGE-1)
    var genWords = generated.split(/\s+/);
    var actWords = actual.split(/\s+/);

    if (genWords.length === 0 || actWords.length === 0) return 0;

    var overlap = 0;
    for (var i = 0; i < genWords.length; i++) {
        if (actWords.indexOf(genWords[i]) !== -1) overlap++;
    }

    var precision = overlap / genWords.length;
    var recall = overlap / actWords.length;

    if (precision + recall === 0) return 0;
    var f1 = (2 * precision * recall) / (precision + recall);
    return Math.round(f1 * 100);
}

function showScore(score) {
    document.getElementById('score-section').style.display = 'block';
    document.getElementById('score-bar').style.width = score + '%';
    document.getElementById('score-value').textContent = score + '% Match';

    var bar = document.getElementById('score-bar');
    if (score >= 60) {
        bar.style.background = 'var(--success)';
    } else if (score >= 30) {
        bar.style.background = 'var(--warning)';
    } else {
        bar.style.background = 'var(--primary)';
    }
}

function clearGenerator() {
    document.getElementById('article-input').value = '';
    document.getElementById('actual-headline').value = '';
    document.getElementById('output-section').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
}

function showError(msg) {
    document.getElementById('error-message').textContent = msg;
    document.getElementById('error-message').style.display = 'block';
}

function copyHeadline() {
    var text = document.getElementById('generated-headline').textContent;
    if (!text) return;

    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function() {
            alert('Headline copied!');
        });
    } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        alert('Headline copied!');
    }
}

// Ctrl+Enter shortcut
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        var articleInput = document.getElementById('article-input');
        if (document.activeElement === articleInput || document.getElementById('page-generator').classList.contains('active')) {
            generateHeadline();
        }
    }
});

// ===== Results =====
async function loadPredictions() {
    var container = document.getElementById('results-container');
    try {
        var res = await fetch(API_BASE + '/predictions');
        var data = await res.json();
        renderPredictions(data.predictions);
    } catch (e) {
        container.innerHTML = '<p class="no-results">Error loading predictions. Is the server running?</p>';
    }
}

function renderPredictions(predictions) {
    var container = document.getElementById('results-container');

    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p class="no-results">No predictions yet. Go to Generator and create some headlines!</p>';
        return;
    }

    var html = '';
    var reversed = predictions.slice().reverse();
    for (var i = 0; i < reversed.length; i++) {
        var pred = reversed[i];
        html += '<div class="result-card">';
        html += '<div class="result-header">';
        html += '<span class="result-number">#' + (predictions.length - i) + '</span>';
        html += '<span class="result-timestamp">' + (pred.timestamp || '') + '</span>';
        html += '</div>';
        html += '<div class="result-headline">📰 ' + (pred.generated_headline || '') + '</div>';
        if (pred.actual_headline) {
            html += '<div class="result-actual">✓ Actual: ' + pred.actual_headline + '</div>';
        }
        html += '<div class="result-preview">' + (pred.article_preview || '') + '</div>';
        html += '</div>';
    }

    container.innerHTML = html;
}

async function clearPredictions() {
    if (!confirm('Clear all prediction history?')) return;
    try {
        await fetch(API_BASE + '/predictions', { method: 'DELETE' });
        loadPredictions();
    } catch (e) {}
}

function exportCSV() {
    window.open(API_BASE + '/predictions/export', '_blank');
}

// ===== Model Info =====
async function loadSystemInfo() {
    var container = document.getElementById('system-info');
    container.innerHTML = '<p class="loading-text">Loading...</p>';
    try {
        var res = await fetch(API_BASE + '/health');
        var data = await res.json();
        container.innerHTML = '';
        container.innerHTML += '<div class="info-row"><span>Status</span><span class="score-good">' + data.status + '</span></div>';
        container.innerHTML += '<div class="info-row"><span>Model Loaded</span><span>' + (data.model_loaded ? '✓ Yes' : '✗ No') + '</span></div>';
        container.innerHTML += '<div class="info-row"><span>Device</span><span>' + data.device + '</span></div>';
        container.innerHTML += '<div class="info-row"><span>Last Check</span><span>' + data.timestamp + '</span></div>';
    } catch (e) {
        container.innerHTML = '<p class="loading-text">Server not reachable.</p>';
    }
}
