<script>
	import { fade, fly } from 'svelte/transition';
	import UploadSection from '$lib/components/upload-section.svelte';
	import MapViewer from '$lib/components/map-viewer.svelte';
	import ResultsPanel from '$lib/components/results-panel.svelte';
	import { predictionStore } from '$lib/stores/prediction-store.js';

	let selectedFile = $state(null);
	let uploadedImageUrl = $state(null);
	let isProcessing = $state(false);

	function handleFileSelected(event) {
		const file = event.detail.file;
		selectedFile = file;

		// Create preview URL
		if (file) {
			uploadedImageUrl = URL.createObjectURL(file);
		}
	}

	async function handlePredict() {
		if (!selectedFile) return;

		isProcessing = true;

		try {
			const formData = new FormData();
			formData.append('image', selectedFile);

			const response = await fetch('http://localhost:5000/api/predict', {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				throw new Error('Prediction failed');
			}

			const data = await response.json();
			predictionStore.setPrediction(data);
		} catch (error) {
			console.error('[v0] Error during prediction:', error);
			alert('Failed to process image. Please ensure the backend server is running.');
		} finally {
			isProcessing = false;
		}
	}

	function handleReset() {
		selectedFile = null;
		uploadedImageUrl = null;
		predictionStore.reset();
	}
</script>

<div class="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-50 via-slate-50 to-white text-slate-800 font-sans selection:bg-blue-100">
	
	<header class="sticky top-0 z-50 border-b border-white/40 bg-white/70 backdrop-blur-md shadow-sm">
		<div class="container mx-auto px-6 py-4">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-4">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 shadow-blue-200 shadow-lg">
						<svg class="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
						</svg>
					</div>
					<div>
						<h1 class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-cyan-600">FloodGuard AI</h1>
						<p class="text-xs text-slate-500 font-medium tracking-wide uppercase">Disaster Segmentation System</p>
					</div>
				</div>
				
				<div class="hidden md:flex items-center gap-3 px-4 py-2 rounded-full bg-white/50 border border-white shadow-sm backdrop-blur-sm">
					<div class="relative flex h-3 w-3">
						<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
						<span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
					</div>
					<span class="text-xs font-semibold text-slate-600">System Online</span>
				</div>
			</div>
		</div>
	</header>

	<main class="container mx-auto px-6 py-10">
		
		<div class="grid gap-8 lg:grid-cols-12">
			
			<div class="lg:col-span-4 space-y-6" in:fly={{ x: -20, duration: 600, delay: 200 }}>
				
				<div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden p-6">
					<h2 class="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
						<svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
						Input Data
					</h2>
					
					<UploadSection
						{selectedFile}
						{isProcessing}
						onfileselect={handleFileSelected}
						onpredict={handlePredict}
						onreset={handleReset}
					/>
				</div>

				{#if $predictionStore.prediction}
					<div in:fly={{ y: 20, duration: 500 }} class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden p-6">
						<h2 class="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
							<svg class="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
							Analysis Report
						</h2>
						<ResultsPanel prediction={$predictionStore.prediction} />
					</div>
				{/if}
			</div>

			<div class="lg:col-span-8 h-full" in:fly={{ x: 20, duration: 600, delay: 300 }}>
				<div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden h-[600px] relative group">
					<div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-cyan-400 to-teal-400 z-10"></div>
					
					{#if !uploadedImageUrl}
						<div class="absolute inset-0 flex flex-col items-center justify-center bg-slate-50 text-slate-400">
							<svg class="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path></svg>
							<p class="text-lg font-medium">Map Visualization</p>
							<p class="text-sm">Upload an image to view flood segmentation overlay</p>
						</div>
					{/if}
					
					<div class="h-full w-full">
						<MapViewer {uploadedImageUrl} prediction={$predictionStore.prediction} />
					</div>
				</div>
			</div>
		</div>

		<div class="mt-16 grid gap-6 md:grid-cols-3">
			<div class="group rounded-2xl bg-white p-6 shadow-md hover:shadow-xl transition-all duration-300 border border-slate-100">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
					<svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
					</svg>
				</div>
				<h3 class="mb-2 text-lg font-bold text-slate-800">Upload Imagery</h3>
				<p class="text-sm text-slate-500 leading-relaxed">
					Support for high-resolution satellite and aerial imagery. Our system automatically pre-processes your data for optimal results.
				</p>
			</div>

			<div class="group rounded-2xl bg-white p-6 shadow-md hover:shadow-xl transition-all duration-300 border border-slate-100">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-teal-50 text-teal-600 group-hover:bg-teal-600 group-hover:text-white transition-colors">
					<svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
					</svg>
				</div>
				<h3 class="mb-2 text-lg font-bold text-slate-800">Deep Learning Analysis</h3>
				<p class="text-sm text-slate-500 leading-relaxed">
					Powered by a custom-trained U-Net architecture that identifies water bodies with high precision and ignores false positives.
				</p>
			</div>

			<div class="group rounded-2xl bg-white p-6 shadow-md hover:shadow-xl transition-all duration-300 border border-slate-100">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-50 text-cyan-600 group-hover:bg-cyan-600 group-hover:text-white transition-colors">
					<svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
				</div>
				<h3 class="mb-2 text-lg font-bold text-slate-800">Risk Assessment</h3>
				<p class="text-sm text-slate-500 leading-relaxed">
					Receive instant feedback on flood severity. The system calculates the percentage of flooded area and assigns a critical risk level.
				</p>
			</div>
		</div>
	</main>

	<footer class="border-t border-slate-200 bg-slate-50 py-12">
		<div class="container mx-auto px-6 text-center">
			<p class="text-sm text-slate-400 font-medium">© 2025 FloodGuard System. Powered by TensorFlow & SvelteKit.</p>
		</div>
	</footer>
</div>