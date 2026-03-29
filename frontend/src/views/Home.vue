<template>
  <div class="simple-home">
    <header class="app-header">
      <h1>Dextora Foresight</h1>
      <p class="powered-tag">Market Prediction &amp; AI Simulation Engine &mdash; Powered by Dextora</p>
      <p>Upload context documents and describe the scenario you want to predict.</p>
    </header>

    <main class="main-content">
      <div class="card">
        <section class="upload-section">
          <h2>1. Context Documents</h2>
          <div 
            class="drop-zone"
            :class="{ 'drag-over': isDragOver, 'has-files': files.length > 0 }"
            @dragover.prevent="handleDragOver"
            @dragleave.prevent="handleDragLeave"
            @drop.prevent="handleDrop"
            @click="triggerFileInput"
          >
            <input
              ref="fileInput"
              type="file"
              multiple
              accept=".pdf,.md,.txt"
              @change="handleFileSelect"
              style="display: none"
              :disabled="loading"
            />
            
            <div v-if="files.length === 0" class="drop-placeholder">
              <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="upload-icon"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
              <p>Drag & drop files or click to browse</p>
              <small>Supported formats: PDF, MD, TXT</small>
            </div>
            
            <div v-else class="file-list">
              <div v-for="(file, index) in files" :key="index" class="file-item">
                <span class="file-name">{{ file.name }}</span>
                <button @click.stop="removeFile(index)" class="remove-btn">
                  ×
                </button>
              </div>
            </div>
          </div>
        </section>

        <section class="prompt-section">
          <h2>2. Prediction Scenario</h2>
          <textarea
            v-model="formData.simulationRequirement"
            class="prompt-input"
            placeholder="Describe what you want to predict or simulate based on the uploaded context..."
            rows="5"
            :disabled="loading"
          ></textarea>
        </section>

        <button 
          class="submit-btn"
          @click="startSimulation"
          :disabled="!canSubmit || loading"
        >
          {{ loading ? 'Starting Engine...' : 'Run Prediction' }}
        </button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const formData = ref({ simulationRequirement: '' })
const files = ref([])
const loading = ref(false)
const isDragOver = ref(false)
const fileInput = ref(null)

const canSubmit = computed(() => {
  return formData.value.simulationRequirement.trim() !== '' && files.value.length > 0
})

const triggerFileInput = () => {
  if (!loading.value) fileInput.value?.click()
}

const handleFileSelect = (event) => {
  const selectedFiles = Array.from(event.target.files)
  addFiles(selectedFiles)
}

const handleDragOver = () => {
  if (!loading.value) isDragOver.value = true
}

const handleDragLeave = () => {
  isDragOver.value = false
}

const handleDrop = (e) => {
  isDragOver.value = false
  if (loading.value) return
  const droppedFiles = Array.from(e.dataTransfer.files)
  addFiles(droppedFiles)
}

const addFiles = (newFiles) => {
  const validFiles = newFiles.filter(file => {
    const ext = file.name.split('.').pop().toLowerCase()
    return ['pdf', 'md', 'txt'].includes(ext)
  })
  files.value.push(...validFiles)
}

const removeFile = (index) => {
  files.value.splice(index, 1)
}

const startSimulation = () => {
  if (!canSubmit.value || loading.value) return
  
  import('../store/pendingUpload.js').then(({ setPendingUpload }) => {
    setPendingUpload(files.value, formData.value.simulationRequirement)
    router.push({
      name: 'Process',
      params: { projectId: 'new' }
    })
  })
}
</script>

<style scoped>
.simple-home {
  min-height: 100vh;
  background-color: #f8fafc;
  font-family: system-ui, -apple-system, sans-serif;
  color: #1e293b;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4rem 2rem;
}

.app-header {
  text-align: center;
  margin-bottom: 3rem;
}

.app-header h1 {
  font-size: 2.5rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.5rem;
}

.app-header p {
  color: #64748b;
  font-size: 1.1rem;
}

.powered-tag {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #2563eb;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 0.2rem 0.85rem;
  margin-bottom: 0.6rem;
}

.main-content {
  width: 100%;
  max-width: 600px;
}

.card {
  background: white;
  border-radius: 12px;
  padding: 2.5rem;
  box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

h2 {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #334155;
}

.upload-section {
  margin-bottom: 2rem;
}

.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background-color: #f8fafc;
}

.drop-zone:hover, .drop-zone.drag-over {
  border-color: #3b82f6;
  background-color: #eff6ff;
}

.drop-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  color: #64748b;
}

.upload-icon {
  margin-bottom: 0.5rem;
  color: #94a3b8;
}

.drop-placeholder small {
  font-size: 0.8rem;
  color: #94a3b8;
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  align-items: flex-start;
}

.file-item {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: white;
  padding: 0.75rem 1rem;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  font-size: 0.9rem;
}

.remove-btn {
  background: none;
  border: none;
  color: #ef4444;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0 0.5rem;
}

.remove-btn:hover {
  text-shadow: 0 0 2px rgba(239, 68, 68, 0.4);
}

.prompt-section {
  margin-bottom: 2rem;
}

.prompt-input {
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 1rem;
  font-family: inherit;
  font-size: 0.95rem;
  resize: vertical;
  background-color: #f8fafc;
  transition: border-color 0.2s;
}

.prompt-input:focus {
  outline: none;
  border-color: #3b82f6;
  background-color: white;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.submit-btn {
  width: 100%;
  background-color: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 1rem;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;
}

.submit-btn:hover:not(:disabled) {
  background-color: #1d4ed8;
}

.submit-btn:disabled {
  background-color: #94a3b8;
  cursor: not-allowed;
}
</style>
