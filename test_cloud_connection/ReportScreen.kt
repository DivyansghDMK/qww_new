package com.deckmount.ecgapp.presentation.ui.screen.report

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import com.deckmount.ecgapp.R
import android.graphics.Color as GColor
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import com.deckmount.ecgapp.presentation.ui.screen.livemonitorecg.ALL_LEADS
import com.deckmount.ecgapp.presentation.ui.screen.livemonitorecg.ECGReportRenderData
import kotlinx.coroutines.sync.Mutex

// ─────────────────────────────────────────────────────────────────────────────
// Page dimension constants (mm)
// ─────────────────────────────────────────────────────────────────────────────

private const val A4_P_W = 210f    // Portrait width
private const val A4_P_H = 297f    // Portrait height
private const val A4_L_W = 297f    // Landscape width
private const val A4_L_H = 210f    // Landscape height

// Printer margins - safe printable area
private const val MARGIN_TOP = 10f     // 10mm top margin
private const val MARGIN_BOTTOM = 10f  // 10mm bottom margin
private const val MARGIN_LEFT = 10f    // 10mm left margin (for landscape)
private const val MARGIN_RIGHT = 10f   // 10mm right margin

// ECG capture constants
private const val ECG_FS = 500f    // Hz
private const val TAG_R = "ECGReportScreen"
private const val ADC_PER_MM = 2.75f  // ← ADD THIS: 640 ADC units per 5mm grid box

// Fixed wave parameters
private const val FIXED_WAVE_SPEED = 25f  // mm/s
private const val FIXED_WAVE_GAIN = 10f   // mm/mV

// ─────────────────────────────────────────────────────────────────────────────
// ECGReportScreen  – main entry composable
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ECGReportScreen(
    reportData: ECGReportRenderData, onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var isExporting by remember { mutableStateOf(false) }
    var isSharing by remember { mutableStateOf(false) }
    var tempPdfFile by remember { mutableStateOf<File?>(null) }

    // Activity result launcher for saving PDF with file picker
    val saveFileLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/pdf")
    ) { uri ->
        uri?.let {
            scope.launch(Dispatchers.IO) {
                try {
                    tempPdfFile?.let { file ->
                        context.contentResolver.openOutputStream(uri)?.use { outputStream ->
                            file.inputStream().use { inputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }
                        // Delete temp file after copying
                        file.delete()

                        withContext(Dispatchers.Main) {
                            Toast.makeText(
                                context,
                                "PDF saved successfully",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG_R, "Error saving PDF: ${e.message}", e)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            context,
                            "Failed to save PDF: ${e.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                } finally {
                    tempPdfFile = null
                    isExporting = false
                }
            }
        } ?: run {
            // User cancelled the file picker
            isExporting = false
            tempPdfFile?.delete()
            tempPdfFile = null
        }
    }

    // Function to share PDF with better compatibility
    @SuppressLint("QueryPermissionsNeeded")
    fun sharePdf(file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                file
            )

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "ECG Report - ${reportData.patientName}")
                putExtra(Intent.EXTRA_TEXT, "Please find the attached ECG report from RhythmPro.")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            // Grant permissions to all possible receivers
            val resInfoList = context.packageManager.queryIntentActivities(
                shareIntent,
                0
            )
            for (resolveInfo in resInfoList) {
                val packageName = resolveInfo.activityInfo.packageName
                context.grantUriPermission(
                    packageName,
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }

            context.startActivity(Intent.createChooser(shareIntent, "Share ECG Report"))

            Toast.makeText(context, "Sharing PDF...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG_R, "Error sharing PDF: ${e.message}", e)
            Toast.makeText(
                context,
                "Failed to share PDF: ${e.message}",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "ECG Report",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        color = Color.Black
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.Black
                        )
                    }
                },
                actions = {
                    // Share Button
                    val shareMutex = remember { Mutex() }

                    IconButton(
                        enabled = !isSharing && !isExporting,
                        onClick = {
                            scope.launch {
                                if (!shareMutex.tryLock()) return@launch

                                try {
                                    isSharing = true

                                    val file = withContext(Dispatchers.IO) {
                                        generateECGPdf(context, reportData)
                                    }

                                    file?.let { sharePdf(it) }

                                } finally {
                                    isSharing = false
                                    shareMutex.unlock()
                                }
                            }
                        }
                    ) {
                        if (isSharing) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(22.dp),
                                color = Color(0xFF6200EE),
                                strokeWidth = 2.dp
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Share,
                                contentDescription = "Share PDF",
                                tint = Color.Black
                            )
                        }
                    }

                    // Download Button
                    IconButton(
                        onClick = {
                            if (isExporting || isSharing) return@IconButton
                            scope.launch {
                                //isExporting = true
                                val file = withContext(Dispatchers.IO) {
                                    generateECGPdf(context, reportData)
                                }
                                if (file != null) {
                                    tempPdfFile = file
                                    // Generate filename with timestamp
                                    val timestamp = SimpleDateFormat(
                                        "yyyyMMdd_HHmmss",
                                        Locale.getDefault()
                                    ).format(Date())
                                    val fileName = "RhythmPro_ECG_$timestamp.pdf"
                                    // Launch file picker
                                    saveFileLauncher.launch(fileName)
                                } else {
                                    //isExporting = false
                                    Toast.makeText(
                                        context,
                                        "PDF generation failed",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                }
                            }
                        }
                    ) {
                        if (isExporting) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(22.dp),
                                color = Color(0xFF6200EE),
                                strokeWidth = 2.dp
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Download,
                                contentDescription = "Download PDF",
                                tint = Color.Black
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.White,
                    titleContentColor = Color.Black,
                    navigationIconContentColor = Color.Black
                )
            )
            HorizontalDivider(
                modifier = Modifier.fillMaxWidth(),
                thickness = 1.dp,
                color = Color.LightGray
            )
        }
    ) { padding ->
        ECGReportCanvas(
            reportData = reportData,
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ECGReportCanvas  – renders bitmap once, then pan + pinch-zoom
// ─────────────────────────────────────────────────────────────────────────────


@Composable
private fun ECGReportCanvas(
    reportData: ECGReportRenderData, modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val density = LocalDensity.current

    val isPortrait = reportData.layout == "1x12"
    val pageW = if (isPortrait) A4_P_W else A4_L_W
    val pageH = if (isPortrait) A4_P_H else A4_L_H

    BoxWithConstraints(modifier = modifier.background(Color(0xFF424242))) {

        val availW = constraints.maxWidth.toFloat()
        val pxPerMm = availW / pageW

        // Display dimensions (ORIGINAL - perfect fit)
        val displayW = availW.toInt().coerceAtLeast(100)
        val displayH = (pageH * pxPerMm).toInt().coerceAtLeast(100)

        //  Bitmap dimensions (LARGER for quality)
        val qualityMultiplier = 2.5f
        val bmpW = (displayW * qualityMultiplier).toInt()
        val bmpH = (displayH * qualityMultiplier).toInt()
        val renderPxPerMm = pxPerMm * qualityMultiplier

        // ── Render the full report once in the background ──────────────────
        val bitmap by produceState<Bitmap?>(
            initialValue = null, key1 = reportData, key2 = bmpW, key3 = bmpH
        ) {
            value = withContext(Dispatchers.Default) {
                try {
                    val bmp = Bitmap.createBitmap(bmpW, bmpH, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(bmp)
                    ECGReportRenderer(reportData, renderPxPerMm, context).draw(canvas)
                    bmp
                } catch (e: Exception) {
                    Log.e(TAG_R, "Bitmap render error: ${e.message}", e)
                    null
                }
            }
        }

        // ── Pan + zoom state ───────────────────────────────────────────────
        var scale by remember { mutableStateOf(1f) }
        var offsetX by remember { mutableStateOf(0f) }
        var offsetY by remember { mutableStateOf(0f) }

        //  Canvas size (ORIGINAL - perfect fit)
        val pageWDp = with(density) { displayW.toFloat().toDp() }
        val pageHDp = with(density) { displayH.toFloat().toDp() }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .clipToBounds()
                .pointerInput(Unit) {
                    detectTransformGestures { _, pan, zoom, _ ->
                        scale = (scale * zoom).coerceIn(0.5f, 8f)
                        offsetX += pan.x
                        offsetY += pan.y
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                androidx.compose.foundation.Canvas(
                    modifier = Modifier
                        .width(pageWDp)
                        .height(pageHDp)
                        .graphicsLayer {
                            scaleX = scale
                            scaleY = scale
                            translationX = offsetX
                            translationY = offsetY
                            transformOrigin = TransformOrigin(0.5f, 0.5f)
                        }) {
                    //THE REAL FIX: Scale bitmap to canvas size explicitly
                    drawImage(
                        image = bitmap!!.asImageBitmap(),
                        dstSize = androidx.compose.ui.unit.IntSize(
                            size.width.toInt(),
                            size.height.toInt()
                        )
                    )
                }
            } else {
                // Loading indicator while bitmap renders
                Box(
                    modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        CircularProgressIndicator(color = Color.White)
                        Text(
                            text = "Rendering ECG report…", color = Color.White, fontSize = 14.sp
                        )
                    }
                }
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// ECGReportRenderer  – all drawing logic on android.graphics.Canvas
//   Used for BOTH in-app bitmap and PDF generation with the same code.
// ─────────────────────────────────────────────────────────────────────────────

internal class ECGReportRenderer(
    private val data: ECGReportRenderData,
    val pxPerMm: Float,
    private val context: Context
) {
    private val isPortrait = data.layout == "1x12"
    val pageW = if (isPortrait) A4_P_W else A4_L_W
    val pageH = if (isPortrait) A4_P_H else A4_L_H

    // Calculate MM_PER_SAMPLE dynamically from FIXED_WAVE_SPEED
    private val mmPerSample =
        FIXED_WAVE_SPEED / ECG_FS  // Always 25/500 = 0.05 mm/sample

    // Whether we are generating a PDF (pxPerMm ≈ 2.83 pt/mm) or rendering for screen (~5+)
    private val isForPdf = pxPerMm < 3.5f

    // ✅ ADD THIS: Scale ADC conversion based on rendering context
    private val adcScaleFactor = if (isForPdf) {
        1.36f  // Base calibration for PDF
    } else {
        4.08f * (A4_P_W / pageW) // Scale up for screen quality multiplier
    }

    // ── Coordinate helpers ─────────────────────────────────────────────────
    /** mm → canvas units (px on screen, pt in PDF) */
    private fun p(mm: Float) = mm * pxPerMm
    //private fun p(mm: Int) = mm.toFloat() * pxPerMm

    /** Convert TCPDF font-pt to canvas units */
    private fun pt(ptSize: Float) = ptSize * 0.352778f * pxPerMm

    // ── Paint factory ─────────────────────────────────────────────────────
    private fun mkText(
        ptSize: Float, bold: Boolean = false, italic: Boolean = false
    ) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(
            "sans-serif", when {
                bold && italic -> Typeface.BOLD_ITALIC
                bold -> Typeface.BOLD
                else -> Typeface.NORMAL
            }
        )
        textSize = pt(ptSize)
        color = GColor.BLACK
    }

    // ── Pre-built paints ──────────────────────────────────────────────────

    private val gridMinor = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.rgb(245, 220, 220)
        strokeWidth = p(0.1f)
    }
    private val gridMajor = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.rgb(230, 150, 150)
        strokeWidth = p(0.25f)
    }
    private val waveP = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.BLACK
        // Guarantee waveform is always visible; PDF can be thinner
        strokeWidth = if (isForPdf) p(0.12f) else p(0.15f).coerceAtLeast(1.0f)
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }
    private val calibP = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.BLACK
        strokeWidth = if (isForPdf) p(0.4f) else p(0.4f).coerceAtLeast(1.5f)
        strokeJoin = Paint.Join.MITER
        strokeCap = Paint.Cap.SQUARE
    }
    private val boxP = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.BLACK
        strokeWidth = if (isForPdf) p(0.3f) else p(0.3f).coerceAtLeast(1.0f)
    }
    private val dashDivP = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = GColor.rgb(80, 80, 80)
        strokeWidth = if (isForPdf) p(0.4f) else p(0.4f).coerceAtLeast(1.2f)
        pathEffect = DashPathEffect(floatArrayOf(p(2f), p(2f)), 0f)
    }

    // Text paints – sizes match SetFont calls
    private val tp9 = mkText(9f)
    private val tp8 = mkText(8f)
    private val tp7 = mkText(7f)
    private val tp6 = mkText(6f)
    private val tp7B = mkText(7f, bold = true)
    private val tp8B = mkText(8f, bold = true)
    private val tp8_5B = mkText(8.5f, bold = true)
    private val tp10B = mkText(10f, bold = true)
    private val tp10_5B = mkText(10.5f, bold = true)
    private val tp12B = mkText(12f, bold = true)
    private val tp12_5B = mkText(12.5f, bold = true)

    // ── Text helper: position like TCPDF Text(x, y, …) top-left ──────────
    /**
     * Draw [text] such that (xMm, yMm) is the top-left of the bounding box –
     * the same semantic as TCPDF's pdf->Text($x, $y, $str).
     */
    private fun drawTxt(
        canvas: Canvas, text: String, xMm: Float, yMm: Float, paint: Paint
    ) {
        val fm = paint.fontMetrics
        // baseline = top + ascent_height; fm.ascent is negative, so: baseline = yPx - fm.ascent
        canvas.drawText(text, p(xMm), p(yMm) - fm.ascent, paint)
    }

    // ─────────────────────────────────────────────────────────────────────
    // PUBLIC – draw everything to [canvas]
    // ─────────────────────────────────────────────────────────────────────

    fun draw(canvas: Canvas) {
        canvas.drawColor(GColor.WHITE)
        drawGrid(canvas, 0f, 0f, pageW, pageH)
        drawHeader(canvas)
        when (data.layout) {
            "1x12" -> draw1x12(canvas)
            "3x4" -> draw3x4(canvas)
            else -> draw2x6(canvas)   // default / "2x6"
        }
        drawFooter(canvas)
    }

    // ─────────────────────────────────────────────────────────────────────
    // ECG GRID  –  minor every 1 mm, major every 5 mm
    // Replicates drawECGGrid() from both files (landscape colour values)
    // ─────────────────────────────────────────────────────────────────────

    private fun drawGrid(canvas: Canvas, x: Float, y: Float, w: Float, h: Float) {
        val wi = w.toInt()
        val hi = h.toInt()

        // Vertical lines
        for (i in 0..wi) {
            val xp = p(x + i)
            val paint = if (i % 5 == 0) gridMajor else gridMinor
            canvas.drawLine(xp, p(y), xp, p(y + h), paint)
        }
        // Horizontal lines
        for (j in 0..hi) {
            val yp = p(y + j)
            val paint = if (j % 5 == 0) gridMajor else gridMinor
            canvas.drawLine(p(x), yp, p(x + w), yp, paint)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // HEADER – patient info  |  measurements  |  extra  |  logo / date
    // ─────────────────────────────────────────────────────────────────────

    private fun drawHeader(canvas: Canvas) {
        val yBase = MARGIN_TOP     // Changed from 5f - 3f (= 2mm) to 10mm
        val lh = 5f          // line height mm
        val leftX = if (isPortrait) 10f else (MARGIN_LEFT + 15f)   // Added margin for landscape

        // ── Column 1: patient ─────────────────────────────────────────────
        var x = leftX
        drawTxt(canvas, "Name: ${data.patientName.ifBlank { "-" }}", x, yBase, tp9)
        drawTxt(canvas, "Age: ${data.patientAge.ifBlank { "-" }}", x, yBase + lh, tp9)
        drawTxt(canvas, "Gender: ${data.patientGender.ifBlank { "-" }}", x, yBase + lh * 2, tp9)
        //drawTxt(canvas, "Weight: ${data.patientWeight.ifBlank { "-" }}", x, yBase + lh * 3, tp9)
        //drawTxt(canvas, "Height: ${data.patientHeight.ifBlank { "-" }}", x, yBase + lh * 4, tp9)

        /*x += 12 * 5f + 2 * 5f   // += 70 mm

        // ── Column 2: ECG measurements ────────────────────────────────────
        drawTxt(canvas, "HR: ${data.hr} bpm", x, yBase, tp9)
        drawTxt(canvas, "PR: ${data.pr} ms", x, yBase + lh, tp9)
        drawTxt(canvas, "QRS: ${data.qrs} ms", x, yBase + lh * 2, tp9)
        drawTxt(canvas, "QT: ${data.qt} ms", x, yBase + lh * 3, tp9)
        drawTxt(canvas, "QTc: ${data.qtc} ms", x, yBase + lh * 4, tp9)

        x += 5 * 5f + 2 * 5f    // += 35 mm*/

        x += if (isPortrait) 40f else (12 * 5f + 2 * 5f)   // Portrait: 50mm spacing, Landscape: 70mm

        // ── Column 2: ECG measurements ────────────────────────────────────
        drawTxt(canvas, "HR: ${data.hr} bpm", x, yBase, tp9)
        drawTxt(canvas, "PR: ${data.pr} ms", x, yBase + lh, tp9)
        drawTxt(canvas, "QRS: ${data.qrs} ms", x, yBase + lh * 2, tp9)
        drawTxt(canvas, "QT: ${data.qt} ms", x, yBase + lh * 3, tp9)
        drawTxt(canvas, "QTc: ${data.qtc} ms", x, yBase + lh * 4, tp9)

        x += if (isPortrait) 35f else (5 * 5f + 2 * 5f)    // Portrait: 40mm spacing, Landscape: 35mm

        // ── Column 3: extra measurements ──────────────────────────────────
        drawTxt(canvas, "RR: ${data.rr} ms", x, yBase, tp9)

        // Format RV5/SV1 to 2 decimal places for precision
        val rv5Str = String.format("%.3f", data.rv5)
        val sv1Str = String.format("%.3f", data.sv1)
        val indexStr = String.format("%.3f", data.rv5 + data.sv1)

        drawTxt(canvas, "RV5/SV1: $rv5Str/$sv1Str mV", x, yBase + lh, tp9)

        // Sokolow-Lyon Index with clinical interpretation
        val indexText = "RV5+SV1: $indexStr mV"
        val indexDisplay = if (data.rv5 + data.sv1 >= 3.5f) {
            // LVH possible - add asterisk to flag elevated values
            indexText + " *"
        } else {
            indexText
        }

        // ★ DISPLAY the Sokolow-Lyon Index (was missing before!)
        drawTxt(canvas, indexDisplay, x, yBase + lh * 2, tp9)
        drawTxt(canvas, "QTcF: ${data.qtcf} ms", x, yBase + lh * 3, tp9)

        // ── Logo (right-aligned, with margin from right edge) ────────────────────
        val logoW = 60f
        val logoH = 10f
        val logoX = pageW - MARGIN_RIGHT - logoW  // Changed from 5f to MARGIN_RIGHT
        val logoY = yBase

        try {
            val bmp = BitmapFactory.decodeResource(context.resources, R.drawable.deck_mount)
            if (bmp != null) {
                val dst = RectF(p(logoX), p(logoY), p(logoX + logoW), p(logoY + logoH))
                canvas.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), dst, null)
            }
        } catch (e: Exception) {
            Log.w(TAG_R, "Logo load failed: ${e.message}")
            // Fallback: draw org name text
            drawTxt(canvas, data.orgName.ifBlank { "Deckmount" }, logoX, logoY + 3f, tp8B)
        }

        // ── Specs row ─────────────────────────────────────────────────────
        val specY = logoY + logoH + 2f
        val specTxt =
            "$FIXED_WAVE_SPEED mm/s   0.5-25 Hz   AC:${data.acFilter}Hz   $FIXED_WAVE_GAIN mm/mV"
        drawTxt(canvas, specTxt, logoX, specY, tp8)

        // ── Date / Time ───────────────────────────────────────────────────
        drawTxt(
            canvas, "Date & Time: ${data.reportDate} ${data.reportTime}", logoX, specY + 4f, tp8
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // 1×12  PORTRAIT LAYOUT
    //   topOffset=38, usableH=237, cellH≈19.75mm
    //      calibration at x=5, label at TOP (labelY = midY - 10), waveform x=25 width=175
    // ─────────────────────────────────────────────────────────────────────

    private fun draw1x12(canvas: Canvas) {
        val topOffset = MARGIN_TOP + 28f  // 10mm margin + 28mm for header = 38mm
        val usableH = A4_P_H - topOffset - MARGIN_BOTTOM - 12f   // Reduced usable height
        val cellH = usableH / 12f

        val samples = data.leadData3500

        ALL_LEADS.forEachIndexed { i, lead ->
            val midY = topOffset + i * cellH + cellH / 2f
            val labelY = midY - 10f

            // 1 mV calibration pulse (portrait: with left margin)
            drawCalibration(canvas, MARGIN_LEFT, midY, FIXED_WAVE_GAIN)

            // Lead label at top
            drawTxt(canvas, lead, MARGIN_LEFT + 11f, labelY, tp8_5B)

            // Waveform: starts with margin
            drawWaveform(
                canvas = canvas,
                samples = samples[lead] ?: emptyList(),
                x0Mm = MARGIN_LEFT + 13f,
                y0Mm = midY,
                widthMm = pageW - MARGIN_LEFT - MARGIN_RIGHT - 15f,  // Full width minus margins
                gainMmPerMv = FIXED_WAVE_GAIN
            )
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 2×6  LANDSCAPE LAYOUT
    //      startY=35, rowH=22, leadW=130 (adjusted for padding), pairs=[I/V1 … aVL/V6]
    //      calibration at x=0(+4 pad), label at x=13, waveform x=18 w=130
    //      dashed divider at x=153 (18+130+5), second lead label at x=158, waveform at x=163 w=130
    //      rhythm strip: midY=176.5, waveform x=25 w=250, USING 3500 SAMPLES
    // ─────────────────────────────────────────────────────────────────────

    private fun draw2x6(canvas: Canvas) {
        val startY = MARGIN_TOP + 25f   // 10mm margin + 25mm for header = 35mm
        val rowH = 22f
        val leadW = 125f  // Slightly reduced to fit with margins
        val divPad = 5f
        val leftMargin = MARGIN_LEFT + 8f  // Add left margin

        val samples = data.leadData1750

        val pairMap = listOf(
            "I" to "V1", "II" to "V2", "III" to "V3", "aVR" to "V4", "aVF" to "V5", "aVL" to "V6"
        )

        pairMap.forEachIndexed { r, (l1, l2) ->
            val midY = startY + r * rowH + rowH / 2f
            val labelY = midY - 10f

            // Calibration with left margin
            drawCalibrationPad(canvas, leftMargin - 4f, midY, FIXED_WAVE_GAIN)

            // ── Left lead ─────────────────────────────────────────────────
            drawTxt(canvas, l1, leftMargin + 9f, labelY, tp10B)
            drawWaveform(
                canvas, samples[l1] ?: emptyList(), leftMargin + 14f, midY, leadW, FIXED_WAVE_GAIN
            )

            // Dashed divider
            val divX = leftMargin + 14f + leadW + divPad
            canvas.drawLine(
                p(divX), p(midY - rowH / 2f), p(divX), p(midY + rowH / 2f), dashDivP
            )

            // ── Right lead with padding ────────────────────────────────────
            val rightLabelX = divX + divPad
            val rightWaveX = rightLabelX + 5f

            drawTxt(canvas, l2, rightLabelX, labelY, tp10B)
            drawWaveform(
                canvas, samples[l2] ?: emptyList(), rightWaveX, midY, leadW, FIXED_WAVE_GAIN
            )
        }

        // ── Rhythm strip ──────────────────────────────────────────────────
        val rhythmMidY = startY + 6f * rowH + 2f + 7.5f
        drawCalibrationPad(canvas, leftMargin - 4f, rhythmMidY, FIXED_WAVE_GAIN)
        drawTxt(canvas, "II", leftMargin + 10f, rhythmMidY - 3f, tp12B)
        drawWaveform(
            canvas = canvas,
            samples = data.leadData5000["II"] ?: emptyList(),
            x0Mm = leftMargin + 21f,
            y0Mm = rhythmMidY,
            widthMm = pageW - leftMargin - MARGIN_RIGHT - 25f,  // Adjust for margins
            gainMmPerMv = FIXED_WAVE_GAIN
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // 3×4  LANDSCAPE LAYOUT
    //      startY=35, rowH=30, leftPad=18, leadW=85 (adjusted), 4 rows × 3 leads
    //      divPad=5mm after each divider
    //      rhythm strip: midY=165.5, waveform x=25 w=250, USING 3500 SAMPLES
    // ─────────────────────────────────────────────────────────────────────

    private fun draw3x4(canvas: Canvas) {
        val startY = MARGIN_TOP + 25f   // 10mm margin + 25mm for header = 35mm
        val rowH = 30f
        val leftMargin = MARGIN_LEFT + 8f
        val leftPad = leftMargin + 10f
        val leadW = 80f  // Adjusted for margins
        val divPad = 5f

        val samples = data.leadData1250

        val leadGroups = listOf(
            listOf("I", "II", "III"),
            listOf("aVR", "aVL", "aVF"),
            listOf("V1", "V2", "V3"),
            listOf("V4", "V5", "V6")
        )

        leadGroups.forEachIndexed { r, group ->
            val midY = startY + r * rowH + rowH / 2f
            val labelY = midY - 10f

            drawCalibrationPad(canvas, leftMargin - 4f, midY, FIXED_WAVE_GAIN)

            group.forEachIndexed { c, lead ->
                val xStart = if (c == 0) {
                    leftPad
                } else {
                    leftPad + c * (leadW + divPad + divPad)
                }

                drawTxt(canvas, lead, xStart, labelY, tp10_5B)
                drawWaveform(
                    canvas, samples[lead] ?: emptyList(), xStart, midY, leadW, FIXED_WAVE_GAIN
                )

                if (c < 2) {
                    val divX = xStart + leadW + divPad
                    canvas.drawLine(
                        p(divX), p(midY - rowH / 2f), p(divX), p(midY + rowH / 2f), dashDivP
                    )
                }
            }
        }

        // ── Rhythm strip ─────────────────────────────────────────────────
        val rhythmMidY = startY + 4f * rowH + 3f + 7.5f
        drawCalibrationPad(canvas, leftMargin - 4f, rhythmMidY, FIXED_WAVE_GAIN)
        drawTxt(canvas, "II", leftMargin + 10f, rhythmMidY - 3f, tp12_5B)
        drawWaveform(
            canvas = canvas,
            samples = data.leadData5000["II"] ?: emptyList(),
            x0Mm = leftMargin + 21f,
            y0Mm = rhythmMidY,
            widthMm = pageW - leftMargin - MARGIN_RIGHT - 25f,
            gainMmPerMv = FIXED_WAVE_GAIN
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // FOOTER
    // ─────────────────────────────────────────────────────────────────────

    private fun drawFooter(canvas: Canvas) {
        if (isPortrait) drawFooterPortrait(canvas) else drawFooterLandscape(canvas)
    }

    // ── Portrait footer ───────────────────────────────────────────────────
    //       footerY=269, doctor at (10,284)/(10,289)
    //       box: x=95,y=272,w=105,h=18   title+conclusions (3 cols)
    //       footer text centred at y=292

    private fun drawFooterPortrait(canvas: Canvas) {
        val footerY = A4_P_H - MARGIN_BOTTOM - 20f  // 10mm margin from bottom

        drawTxt(canvas, "Doctor Name: ", MARGIN_LEFT, footerY + 15f, tp8)
        drawTxt(canvas, "Doctor Sign: ", MARGIN_LEFT, footerY + 20f, tp8)

        // Conclusion box
        val boxX = 95f
        val boxY = footerY + 3f
        val boxW = pageW - boxX - MARGIN_RIGHT - 5f  // Adjust for right margin
        val boxH = 18f
        canvas.drawRect(p(boxX), p(boxY), p(boxX + boxW), p(boxY + boxH), boxP)

        // Title – centred in box
        val titleX = boxX + (boxW - tp7B.measureText("CONCLUSION") / pxPerMm) / 2f
        drawTxt(canvas, "CONCLUSION", titleX, boxY + 1f, tp7B)

        // Conclusion items: 3 columns
        val cols = 3
        val colW = (boxW - 4f) / cols
        val rowH = 3.5f
        val startX = boxX + 2f
        val startY = boxY + 6f

        data.conclusions.forEachIndexed { i, line ->
            val row = i / cols
            val col = i % cols
            val tx = startX + col * colW
            val ty = startY + row * rowH
            if (ty + rowH > boxY + boxH) return@forEachIndexed
            drawTxt(canvas, "${i + 1}. $line", tx, ty, tp6)
        }

        // Footer text centred
        val fTxt = "Deckmount Electronics Pvt Ltd | RhythmPro ECG | IEC 60601 | Made in India"
        val fW = tp7.measureText(fTxt) / pxPerMm
        drawTxt(canvas, fTxt, (pageW - fW) / 2f, A4_P_H - MARGIN_BOTTOM + 2f, tp7)
    }

    private fun drawFooterLandscape(canvas: Canvas) {
        val footerTopY = A4_L_H - MARGIN_BOTTOM - 12f

        drawTxt(canvas, "Doctor Name: ", MARGIN_LEFT + 3f, footerTopY + 5f, tp8)
        drawTxt(canvas, "Doctor Sign: ", MARGIN_LEFT + 3f, footerTopY + 10f, tp8)

        // Conclusion box (right side of page with margin)
        val boxW = 145f
        val boxH = 20f
        val boxX = A4_L_W - boxW - MARGIN_RIGHT
        val boxY = footerTopY - 5f
        canvas.drawRect(p(boxX), p(boxY), p(boxX + boxW), p(boxY + boxH), boxP)

        // Title – centred in box
        val titlePaint = mkText(8f, bold = true)
        val titleX = boxX + (boxW - titlePaint.measureText("CONCLUSION") / pxPerMm) / 2f
        drawTxt(canvas, "CONCLUSION", titleX, boxY + 2f - 2f, titlePaint)

        // Conclusion items: 3 columns
        val cols = 3
        val colGap = 5f
        val colW = (boxW - 10f - colGap * 2f) / cols
        val rowGap = 5f
        val startX = boxX + 5f
        val startY = boxY + 8f - 3f

        var sr = 1
        data.conclusions.forEachIndexed { i, txt ->
            val row = i / cols
            val col = i % cols
            val tx = startX + col * (colW + colGap)
            val ty = startY + row * rowGap
            if (ty + rowGap > boxY + boxH - 1f) return@forEachIndexed
            drawTxt(canvas, "$sr. $txt", tx, ty, tp6)
            sr++
        }

        // Footer text centred
        val fTxt = "Deckmount Electronics Pvt Ltd | RhythmPro ECG | IEC 60601 | Made in India"
        val fW = tp8.measureText(fTxt) / pxPerMm
        drawTxt(canvas, fTxt, (pageW - fW) / 2f, A4_L_H - MARGIN_BOTTOM + 4f, tp8)
    }

    // ── Landscape footer ──────────────────────────────────────────────────
    //      footerTopY=190, doctor at (8,195)/(8,200)
    //       box: x=144,y=185,w=150,h=20   title+conclusions (3 cols)
    //       footer text centred at y=206


    // ─────────────────────────────────────────────────────────────────────
    // CALIBRATION PULSE  (1 mV square wave)
    //
    //   Portrait  (drawCalibration):
    //     Line(x, y → x+2, y)           ← baseline 2 mm
    //     Line(x+2, y → x+2, y-gain)    ← rise
    //     Line(x+2, y-gain → x+7, y-gain) ← top 5 mm
    //     Line(x+7, y-gain → x+7, y)    ← fall
    //     Line(x+7, y → x+9, y)         ← baseline 2 mm
    //
    //   Landscape  adds pad=4 mm before the pulse.
    // ─────────────────────────────────────────────────────────────────────

    /** Portrait-style calibration (no left padding). */
    private fun drawCalibration(
        canvas: Canvas, xMm: Float, yMm: Float, gainMm: Float
    ) {
        val x = p(xMm)
        val y = p(yMm)
        val g = p(gainMm)

        canvas.drawLine(x, y, x + p(2f), y, calibP)
        canvas.drawLine(x + p(2f), y, x + p(2f), y - g, calibP)
        canvas.drawLine(x + p(2f), y - g, x + p(7f), y - g, calibP)
        canvas.drawLine(x + p(7f), y - g, x + p(7f), y, calibP)
        canvas.drawLine(x + p(7f), y, x + p(9f), y, calibP)
    }

    /*private fun drawCalibration(
        canvas: Canvas, xMm: Float, yMm: Float, gainMm: Float
    ) {
        val x = p(xMm)
        val y = p(yMm)
        val g = p(gainMm)

        // Left bottom line removed
        canvas.drawLine(x + p(2f), y, x + p(2f), y - g, calibP)
        canvas.drawLine(x + p(2f), y - g, x + p(7f), y - g, calibP)
        canvas.drawLine(x + p(7f), y - g, x + p(7f), y, calibP)
        canvas.drawLine(x + p(7f), y, x + p(9f), y, calibP)
    }*/

    /** Landscape-style calibration: 4 mm left pad then same pulse. */
    private fun drawCalibrationPad(
        canvas: Canvas, xMm: Float, yMm: Float, gainMm: Float
    ) {
        drawCalibration(canvas, xMm + 4f, yMm, gainMm)
    }

    // ─────────────────────────────────────────────────────────────────────
    // WAVEFORM  –  draws mV samples as line segments
    //
    //    equivalent:
    //     mmPerSample = speed / fs              (0.05 mm/sample)
    //     y1 = y0 - (sample / 1 mV) * gain_mm  ← we already have mV
    //
    //   Android:
    //     xPx = p(x0 + i * mmPerSample)
    //     yPx = p(y0) - sample_mV * gainMmPerMv * pxPerMm
    // ─────────────────────────────────────────────────────────────────────

    private fun drawWaveform(
        canvas: Canvas,
        samples: List<Float>,
        x0Mm: Float,
        y0Mm: Float,
        widthMm: Float,
        gainMmPerMv: Float
    ) {
        if (samples.size < 2) return

        val maxXMm = x0Mm + widthMm
        val y0Px = p(y0Mm)

        var prevXPx = p(x0Mm)
        var prevYPx = y0Px - samples[0] * gainMmPerMv * adcScaleFactor  // Use scaled factor

        for (i in 1 until samples.size) {
            val xMm = x0Mm + i * mmPerSample
            if (xMm > maxXMm) break

            val xPx = p(xMm)
            val yPx = y0Px - samples[i] * gainMmPerMv * adcScaleFactor  // Use scaled factor

            canvas.drawLine(prevXPx, prevYPx, xPx, yPx, waveP)
            prevXPx = xPx
            prevYPx = yPx
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF GENERATION  –  uses android.graphics.pdf.PdfDocument (A4 in PDF points)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Renders the ECG report to a PDF file and returns it.
 * A4 dimensions in PDF points: portrait 595×842, landscape 842×595.
 * 1 mm = 2.83465 pt, so pxPerMm ≈ 2.83 (the same rendering code is reused).
 */
fun generateECGPdf(context: Context, data: ECGReportRenderData): File? {
    return try {
        val isPortrait = data.layout == "1x12"

        // A4 in PDF points (1 pt = 1/72 inch = 0.352778 mm)
        val pageWPt = if (isPortrait) 595 else 842
        val pageHPt = if (isPortrait) 842 else 595
        val pageWMm = if (isPortrait) A4_P_W else A4_L_W

        // Points per mm – matches TCPDF's internal coordinate system
        val pxPerMm = pageWPt.toFloat() / pageWMm   // ≈ 2.833 pt/mm

        val document = PdfDocument()
        val pageInfo = PdfDocument.PageInfo.Builder(pageWPt, pageHPt, 1).create()
        val page = document.startPage(pageInfo)

        ECGReportRenderer(data, pxPerMm, context).draw(page.canvas)

        document.finishPage(page)

        // ── Save to cache directory for sharing (better for FileProvider) ────
        val cacheDir = File(context.cacheDir, "shared_pdfs")
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }

        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val file = File(cacheDir, "RhythmPro_ECG_$ts.pdf")

        FileOutputStream(file).use { fos -> document.writeTo(fos) }
        document.close()

        Log.d(TAG_R, "PDF saved → ${file.absolutePath}")
        file

    } catch (e: Exception) {
        Log.e(TAG_R, "PDF generation failed: ${e.message}", e)
        null
    }
}