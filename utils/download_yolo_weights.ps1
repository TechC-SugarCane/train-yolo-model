param(
    [string]$ModelVersion,
    [string[]]$WeightNameList
)

$WeightDir = ".\weights\"
$WeightMapPath = Join-Path -Path $WeightDir -ChildPath 'weights_map.json'
# jsonファイルを読み込む
$WEIGHT_MAP = Get-Content -Raw -Path $WeightMapPath | ConvertFrom-Json -AsHashtable

$YOLO_NAME = $WEIGHT_MAP.$ModelVersion.name
$YOLO_WEIGHTS_LIST = $WEIGHT_MAP.$ModelVersion.weights
$YOLO_URL = $WEIGHT_MAP.$ModelVersion.url

# 指定された重みファイルをダウンロードする関数
function Download-YOLOWeight {
    param(
        [string]$ModelName,
        [string]$ModelUrl,
        [string]$ModelVersion,
        [string]$WeightDir,
        [string]$WeightName
    )
    # 重みファイル名を作成
    $FullWeightName = "$ModelName$WeightName.pt"
    $WeightsUrl = "$ModelUrl$FullWeightName"
    $SaveDir = Join-Path -Path $WeightDir -ChildPath $ModelVersion
    $WeightsPath = Join-Path -Path $SaveDir -ChildPath $FullWeightName
    $WeightsPathTmp = "$WeightsPath.tmp"

    if (Test-Path $WeightsPath) {
        Write-Host "重みファイル '$FullWeightName' は既に存在します。"
        $response = Read-Host "再ダウンロードしますか？ (y/n) [n]"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Remove-Item $WeightsPath -Force
        } else {
            return
        }
    }

    # 保存先ディレクトリが存在しない場合は作成
    if (-not (Test-Path $SaveDir)) {
        New-Item -ItemType Directory -Path $SaveDir | Out-Null
    }

    Write-Host "重みファイルをダウンロード中: $WeightsUrl -> $WeightsPath"
    try {
        Invoke-WebRequest -Uri $WeightsUrl -OutFile $WeightsPathTmp -ErrorAction Stop
        Move-Item -Path $WeightsPathTmp -Destination $WeightsPath -Force
        Write-Host "ダウンロード完了: $WeightsPath"
    }
    catch {
        Write-Error "ダウンロード中にエラーが発生しました: $_"
    }
    finally {
        # 一時ファイルが存在する場合は削除
        if (Test-Path $WeightsPathTmp) {
            Remove-Item $WeightsPathTmp -Force
        }
    }
}

# 重み名の妥当性を確認する関数
function Test-ValidWeightName {
    param(
        [string[]]$AvailableWeightNames,
        [string]$WeightName
    )
    if ($AvailableWeightNames -contains $WeightName) {
        Write-Host "有効な重み名を確認: $WeightName"
        return $true
    } else {
        Write-Error "無効な重み名: $WeightName. 利用可能な重み名: $($AvailableWeightNames -join ', ')"
        return $false
    }
}

# 指定されたYOLOモデルの重みを処理する関数
function Invoke-YOLO {
    param(
        [hashtable]$WEIGHT_MAP,
        [string]$ModelVersion,
        [string[]]$WeightNameList,
        [string]$WeightDir,
        [string]$YOLO_NAME,
        [string]$YOLO_URL,
        [string[]]$YOLO_WEIGHTS_LIST
    )

    $modelKey = $ModelVersion.ToLower()
    if ($WEIGHT_MAP.ContainsKey($modelKey)) {
        Write-Host "モデルバージョン: $ModelVersion"
        foreach ($weightName in $WeightNameList) {
            Write-Host "------------------------------------------------------------"
            if (Test-ValidWeightName -AvailableWeightNames $YOLO_WEIGHTS_LIST -WeightName $weightName) {
                Download-YOLOWeight -WeightDir $WeightDir -ModelName $YOLO_NAME -ModelUrl $YOLO_URL -ModelVersion $modelKey -WeightName $weightName
            }
        }
    } else {
        Write-Error "無効なモデルバージョン: $ModelVersion"
        Write-Error "利用可能なモデルバージョン: $($WEIGHT_MAP.Keys -join ', ')"
    }
}

# メイン実行部（スプラッティングにより、引数が増えても管理しやすくする）
$InvokeParams = @{
    WEIGHT_MAP        = $WEIGHT_MAP
    ModelVersion      = $ModelVersion
    WeightNameList    = $WeightNameList
    WeightDir         = $WeightDir
    YOLO_NAME         = $YOLO_NAME
    YOLO_URL          = $YOLO_URL
    YOLO_WEIGHTS_LIST = $YOLO_WEIGHTS_LIST
}

Invoke-YOLO @InvokeParams
