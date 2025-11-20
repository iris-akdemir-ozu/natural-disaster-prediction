<script>
	import { onMount } from 'svelte';
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

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50">
	<!-- Header -->
	<header class="border-b border-border bg-background/80 backdrop-blur-sm">
		<div class="container mx-auto px-4 py-6">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-3">
					<div class="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-600">
						<svg
							class="h-7 w-7 text-white"
							fill="none"
							stroke="currentColor"
							viewBox="0 0 24 24"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
							/>
						</svg>
					</div>
					<div>
						<h1 class="text-2xl font-bold text-foreground">Flood Segmentation System</h1>
						<p class="text-sm text-muted-foreground">AI-Powered Natural Disaster Prediction</p>
					</div>
				</div>
				<div class="flex items-center gap-2">
					<div
						class="flex items-center gap-2 rounded-full bg-green-100 px-3 py-1.5 text-sm font-medium text-green-700"
					>
						<span class="h-2 w-2 rounded-full bg-green-600"></span>
						Online
					</div>
				</div>
			</div>
		</div>
	</header>

	<!-- Main Content -->
	<main class="container mx-auto px-4 py-8">
		<div class="grid gap-6 lg:grid-cols-3">
			<!-- Left Panel - Upload & Controls -->
			<div class="space-y-6 lg:col-span-1">
				<UploadSection
					{selectedFile}
					{isProcessing}
					onfileselect={handleFileSelected}
					onpredict={handlePredict}
					onreset={handleReset}
				/>

				{#if $predictionStore.prediction}
					<ResultsPanel prediction={$predictionStore.prediction} />
				{/if}
			</div>

			<!-- Right Panel - Map & Visualization -->
			<div class="lg:col-span-2">
				<MapViewer {uploadedImageUrl} prediction={$predictionStore.prediction} />
			</div>
		</div>

		<!-- Info Section -->
		<div class="mt-12 grid gap-6 md:grid-cols-3">
			<div class="rounded-xl border border-border bg-card p-6">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
					<svg class="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
						/>
					</svg>
				</div>
				<h3 class="mb-2 font-semibold text-card-foreground">Upload Image</h3>
				<p class="text-sm text-muted-foreground">
					Upload satellite or aerial imagery to analyze flood risk and affected areas.
				</p>
			</div>

			<div class="rounded-xl border border-border bg-card p-6">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-teal-100">
					<svg class="h-6 w-6 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
						/>
					</svg>
				</div>
				<h3 class="mb-2 font-semibold text-card-foreground">AI Analysis</h3>
				<p class="text-sm text-muted-foreground">
					U-Net deep learning model analyzes the image and identifies flood-affected regions.
				</p>
			</div>

			<div class="rounded-xl border border-border bg-card p-6">
				<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-cyan-100">
					<svg class="h-6 w-6 text-cyan-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
						/>
					</svg>
				</div>
				<h3 class="mb-2 font-semibold text-card-foreground">View Results</h3>
				<p class="text-sm text-muted-foreground">
					Interactive map visualization shows flood zones with risk levels and confidence scores.
				</p>
			</div>
		</div>
	</main>

	<!-- Footer -->
	<footer class="mt-16 border-t border-border bg-background/50 py-8">
		<div class="container mx-auto px-4 text-center text-sm text-muted-foreground">
			<p>Flood Segmentation System powered by U-Net Deep Learning</p>
			<p class="mt-2">For educational and research purposes</p>
		</div>
	</footer>
</div>
