import 'dart:io';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:image_picker/image_picker.dart';
import 'media_model.dart';
import 'media_repository.dart';

class MediaGalleryScreen extends StatefulWidget {
  const MediaGalleryScreen({super.key});

  @override
  State<MediaGalleryScreen> createState() => _MediaGalleryScreenState();
}

class _MediaGalleryScreenState extends State<MediaGalleryScreen> {
  final MediaRepository _repo = MediaRepository();
  final ImagePicker _picker = ImagePicker();

  Future<void> _handleMediaPick(bool isVideo, ImageSource source) async {
    try {
      final XFile? xFile =
          isVideo
              ? await _picker.pickVideo(source: source)
              : await _picker.pickImage(source: source);

      if (xFile != null) {
        final File file = File(xFile.path);
        await _repo.uploadExistingMedia(file, isVideo ? 'video' : 'photo');
      }
    } catch (e) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(
        // ignore: use_build_context_synchronously
        context,
      ).showSnackBar(SnackBar(content: Text('Error: ${e.toString()}')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Leak Documentation'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed:
                () => showModalBottomSheet(
                  context: context,
                  builder:
                      (_) => SafeArea(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            _buildUploadOption(
                              Icons.camera,
                              'Take Photo',
                              false,
                              ImageSource.camera,
                            ),
                            _buildUploadOption(
                              Icons.videocam,
                              'Record Video',
                              true,
                              ImageSource.camera,
                            ),
                            _buildUploadOption(
                              Icons.photo,
                              'From Gallery',
                              false,
                              ImageSource.gallery,
                            ),
                            _buildUploadOption(
                              Icons.video_library,
                              'Video from Gallery',
                              true,
                              ImageSource.gallery,
                            ),
                          ],
                        ),
                      ),
                ),
          ),
        ],
      ),
      body: StreamBuilder<List<MediaItem>>(
        stream: _repo.getMediaStream(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }

          final mediaItems = snapshot.data!;

          if (mediaItems.isEmpty) {
            return Center(
              child: Text(
                'No leak documentation found',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            );
          }

          return GridView.builder(
            padding: const EdgeInsets.all(8),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 8,
              mainAxisSpacing: 8,
              childAspectRatio: 0.8,
            ),
            itemCount: mediaItems.length,
            itemBuilder:
                (context, index) => _MediaTile(
                  item: mediaItems[index],
                  onDelete: () => _repo.deleteMedia(mediaItems[index].id!),
                ),
          );
        },
      ),
    );
  }

  ListTile _buildUploadOption(
    IconData icon,
    String text,
    bool isVideo,
    ImageSource source,
  ) {
    return ListTile(
      leading: Icon(icon),
      title: Text(text),
      onTap: () {
        Navigator.pop(context);
        _handleMediaPick(isVideo, source);
      },
    );
  }
}

class _MediaTile extends StatelessWidget {
  final MediaItem item;
  final VoidCallback onDelete;

  const _MediaTile({required this.item, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _showDetails(context),
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Stack(
          children: [
            item.type == 'photo'
                ? _buildImage()
                : _VideoPreview(fileUrl: item.fileUrl),
            Positioned(
              right: 8,
              top: 8,
              child: IconButton(
                icon: const Icon(Icons.delete, color: Colors.red),
                onPressed: onDelete,
              ),
            ),
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black54,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (item.description != null)
                      Text(
                        item.description!,
                        maxLines: 1,
                        style: const TextStyle(color: Colors.white),
                      ),
                    Text(
                      item.formattedDate,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImage() {
    return CachedNetworkImage(
      imageUrl: item.fileUrl,
      fit: BoxFit.cover,
      placeholder: (_, __) => const Center(child: CircularProgressIndicator()),
      errorWidget: (_, __, ___) => const Icon(Icons.error),
    );
  }

  void _showDetails(BuildContext context) {
    showDialog(
      context: context,
      builder:
          (_) => AlertDialog(
            title: const Text('Leak Documentation Details'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Type: ${item.type.toUpperCase()}'),
                Text('Date: ${item.formattedDate}'),
                if (item.location != null) Text('Location: ${item.location}'),
                const SizedBox(height: 16),
                SizedBox(
                  height: 300,
                  child:
                      item.type == 'photo'
                          ? _buildImage()
                          : _VideoPlayer(fileUrl: item.fileUrl),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
            ],
          ),
    );
  }
}

class _VideoPreview extends StatefulWidget {
  final String fileUrl;

  const _VideoPreview({required this.fileUrl});

  @override
  State<_VideoPreview> createState() => _VideoPreviewState();
}

class _VideoPreviewState extends State<_VideoPreview> {
  late VideoPlayerController _controller;

  @override
  void initState() {
    super.initState();
    // ignore: deprecated_member_use
    _controller = VideoPlayerController.network(widget.fileUrl)
      ..initialize().then((_) => setState(() {}));
  }

  @override
  Widget build(BuildContext context) {
    return _controller.value.isInitialized
        ? Stack(
          alignment: Alignment.center,
          children: [
            VideoPlayer(_controller),
            const Icon(
              Icons.play_circle_filled,
              size: 40,
              color: Colors.white70,
            ),
          ],
        )
        : const Center(child: CircularProgressIndicator());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class _VideoPlayer extends StatefulWidget {
  final String fileUrl;

  const _VideoPlayer({required this.fileUrl});

  @override
  State<_VideoPlayer> createState() => _VideoPlayerState();
}

class _VideoPlayerState extends State<_VideoPlayer> {
  late VideoPlayerController _controller;
  bool _playing = false;

  @override
  void initState() {
    super.initState();
    // ignore: deprecated_member_use
    _controller = VideoPlayerController.network(widget.fileUrl)
      ..initialize().then((_) => setState(() {}));
  }

  @override
  Widget build(BuildContext context) {
    return _controller.value.isInitialized
        ? GestureDetector(
          onTap:
              () => setState(() {
                _playing ? _controller.pause() : _controller.play();
                _playing = !_playing;
              }),
          child: Stack(
            alignment: Alignment.center,
            children: [
              VideoPlayer(_controller),
              Icon(
                _playing ? Icons.pause : Icons.play_arrow,
                size: 50,
                color: Colors.white70,
              ),
            ],
          ),
        )
        : const Center(child: CircularProgressIndicator());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
